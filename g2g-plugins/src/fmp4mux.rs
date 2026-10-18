//! Pure fragmented-MP4 / CMAF box writer (M24, HEVC in M31): the fMP4 muxing
//! state machine (`Fmp4Muxer`) plus the NAL / `avcC` / `hvcC` helpers shared
//! across the container elements. Annex-B H.264/H.265 access units in, an
//! `ftyp`+`moov` init segment then one `moof`+`mdat` fragment per access unit
//! out, so the recording is playable (ffplay/VLC/browsers via MSE) and durable:
//! a truncated live recording stays valid up to the last complete fragment.
//!
//! The `moov` needs the stream's parameter sets, which arrive in-band with the
//! first IDR, so the init segment is emitted on the first access unit; H.264
//! carries `avc1`/`avcC` with SPS+PPS, H.265 `hvc1`/`hvcC` with VPS+SPS+PPS.
//! Samples are AVCC-style 4-byte length-prefixed NALUs. One fragment per access
//! unit favours latency/durability over container overhead (~100 bytes/frame).
//!
//! Wrapped by [`Mp4Mux`](crate::mp4mux) / the A/V [`Mp4MuxN`](crate::mp4muxn)
//! (which forward the bytes to any sink, e.g. `mp4mux ! filesink`); the helpers
//! are reused by the Matroska A/V muxer [`MkvMuxN`](crate::mkvmuxn). Pure
//! `no_std + alloc`.

use alloc::vec::Vec;

use g2g_core::{Chapter, G2gError, TagList, VideoCodec};

use g2g_core::time::ns_to_ticks;

use crate::annexb::au_is_keyframe;
use crate::mp4muxn::{
    av_fragment, av_moov, FragmentSample, MoovMetadata, PendingSample, TrackInit, VIDEO_TIMESCALE,
};
/// Fallback per-frame duration when the stream carries no timing: 1/30 s.
const DEFAULT_DURATION_NS: u64 = 33_333_333;

/// Pure fragmented-MP4 box writer: the muxing state machine wrapped by
/// [`crate::mp4mux::Mp4Mux`] (forwards the bytes downstream, e.g. to a
/// `filesink`) and the A/V [`crate::mp4muxn::Mp4MuxN`]. Annex-B H.264/H.265
/// access units in, an `ftyp`+`moov` init segment then one `moof`+`mdat`
/// fragment per AU out. The init segment is emitted on the first
/// [`push_au`](Self::push_au) because the `moov` needs the stream's in-band
/// parameter sets.
#[derive(Debug)]
pub(crate) struct Fmp4Muxer {
    codec: VideoCodec,
    width: u32,
    height: u32,
    tags: TagList,
    /// The table of contents, written as the `moov`'s `udta/chpl` (M1046).
    chapters: Vec<Chapter>,
    header_written: bool,
    /// `mfhd` sequence number of the last `moof` written. A chunk is a `moof`,
    /// so this counts chunks, not fragments, once chunking is on.
    sequence: u64,
    /// Accumulated decode time in media-timescale units (`tfdt`).
    decode_time: u64,
    /// Decode-order timestamp of the previous access unit (its DTS when
    /// upstream set one, else its PTS).
    prev_decode_ns: Option<u64>,
    /// One frame's duration from the caps framerate, the fallback for a sample
    /// upstream did not time (an encoder's first frame).
    frame_ns: u64,
    /// Target fragment duration in nanoseconds (`0` = one access unit per fragment,
    /// the default). When set, access units are batched into a single multi-sample
    /// `moof`+`mdat` until the accumulated duration reaches the target, closing the
    /// fragment at the next sync sample (so each fragment begins at a keyframe, the
    /// CMAF / DASH segment shape). Cuts per-AU box overhead for low-fps or
    /// high-bitrate streams.
    fragment_duration_ns: u64,
    /// CMAF conformance mode (M832): CMAF brands on the `ftyp`, a `styp` opening
    /// every segment, and fragments that only ever start on a sync sample (so a
    /// `fragment_duration_ns` of 0 means one fragment per GOP, not per access
    /// unit). See [`with_cmaf`](Self::with_cmaf).
    cmaf: bool,
    /// Target CMAF chunk duration in nanoseconds (`0` = no chunking, the whole
    /// fragment is one `moof`+`mdat`). See [`with_chunk_duration_ns`](Self::with_chunk_duration_ns).
    chunk_duration_ns: u64,
    /// Producer wall clock (64-bit NTP) for the `prft`, injected by the element
    /// so this writer makes no syscall of its own. `None` writes no `prft`.
    producer_clock: Option<fn() -> u64>,
    /// Samples buffered for the open chunk (batched mode); the whole open
    /// fragment when chunking is off.
    pending: Vec<PendingSample>,
    /// `decode_time` at the start of the open chunk (its `tfdt`).
    pending_decode_time: u64,
    /// Accumulated wall duration of the open chunk, in nanoseconds.
    pending_dur_ns: u64,
    /// Accumulated wall duration of the open fragment (every chunk of it so far
    /// plus `pending`), which is what the fragment target is measured against.
    fragment_dur_ns: u64,
    /// Whether a chunk of the current fragment has already been emitted, i.e. the
    /// fragment's `styp` / `prft` are written and the next chunk continues it.
    fragment_started: bool,
    /// Producer wall clock sampled when the open fragment took its first sample,
    /// written in that fragment's `prft`.
    fragment_ntp: u64,
}

impl Fmp4Muxer {
    pub(crate) fn new(codec: VideoCodec, width: u32, height: u32, tags: TagList) -> Self {
        Self {
            codec,
            width,
            height,
            tags,
            chapters: Vec::new(),
            header_written: false,
            sequence: 0,
            decode_time: 0,
            prev_decode_ns: None,
            frame_ns: DEFAULT_DURATION_NS,
            fragment_duration_ns: 0,
            cmaf: false,
            chunk_duration_ns: 0,
            producer_clock: None,
            pending: Vec::new(),
            pending_decode_time: 0,
            pending_dur_ns: 0,
            fragment_dur_ns: 0,
            fragment_started: false,
            fragment_ntp: 0,
        }
    }

    /// Attach the table of contents, written as the `moov`'s `udta/chpl`.
    pub(crate) fn with_chapters(mut self, chapters: Vec<Chapter>) -> Self {
        self.chapters = chapters;
        self
    }

    /// Batch access units into fragments of at least `ns` (closed at the next sync
    /// sample); `0` keeps one fragment per AU. See [`fragment_duration_ns`](Self::fragment_duration_ns).
    pub(crate) fn with_fragment_duration_ns(mut self, ns: u64) -> Self {
        self.fragment_duration_ns = ns;
        self
    }

    /// Write a CMAF (ISO/IEC 23000-19) track file: the `ftyp` carries the `cmfc`
    /// structural brand, every fragment opens a CMAF segment with its own `styp`,
    /// and a fragment never starts mid-GOP (a CMAF fragment must begin at a stream
    /// access point). The rest of the layout already conforms: one track, `mvex` /
    /// `trex`, a `tfdt` in every `traf`, `default-base-is-moof`, and no `mdat`
    /// outside a fragment.
    pub(crate) fn with_cmaf(mut self, cmaf: bool) -> Self {
        self.cmaf = cmaf;
        self
    }

    /// Split each fragment into CMAF chunks of at least `ns` (M859): a chunk is a
    /// `moof`+`mdat` over the samples buffered so far, emitted as soon as it
    /// reaches the target instead of at the end of the fragment, which is what
    /// takes the muxer's latency contribution from a fragment to a chunk. `0`
    /// (the default) keeps one `moof`+`mdat` per fragment. Only the batched modes
    /// have a fragment to subdivide, so this is inert with
    /// `fragment_duration_ns == 0` and CMAF off.
    pub(crate) fn with_chunk_duration_ns(mut self, ns: u64) -> Self {
        self.chunk_duration_ns = ns;
        self
    }

    /// Write a `prft` ahead of each fragment mapping its first sample's decode
    /// time to the producer's wall clock, so a player can measure end-to-end
    /// latency (M859). The clock is injected (a 64-bit NTP timestamp) because
    /// this writer is the `no_std` half.
    pub(crate) fn with_producer_clock(mut self, clock: Option<fn() -> u64>) -> Self {
        self.producer_clock = clock;
        self
    }

    /// Apply a mid-stream caps refinement. A geometry change before the header is
    /// written is absorbed; a codec swap after the `moov` is written is not
    /// expressible and fails loud.
    pub(crate) fn update_caps(
        &mut self,
        codec: VideoCodec,
        width: u32,
        height: u32,
    ) -> Result<(), G2gError> {
        if self.header_written && codec != self.codec {
            return Err(G2gError::CapsMismatch);
        }
        self.codec = codec;
        if width != 0 {
            self.width = width;
        }
        if height != 0 {
            self.height = height;
        }
        Ok(())
    }

    /// Mux one Annex-B access unit, returning the bytes to emit. The first call
    /// prepends the `ftyp`+`moov` init segment. `duration_ns` of 0 derives the
    /// sample duration from the PTS delta (or a default for the first frame).
    /// A frame's nominal duration from the caps framerate, used when upstream
    /// does not time a sample and no previous sample gives a delta.
    pub(crate) fn with_frame_duration_ns(mut self, ns: u64) -> Self {
        if ns > 0 {
            self.frame_ns = ns;
        }
        self
    }

    pub(crate) fn push_au(
        &mut self,
        annexb: &[u8],
        pts_ns: u64,
        dts_ns: u64,
        duration_ns: u64,
    ) -> Result<Vec<u8>, G2gError> {
        let nalus = split_annexb(annexb);
        if nalus.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        let mut out = Vec::new();
        if !self.header_written {
            let param_sets = parameter_sets(self.codec, &nalus)?;
            out.extend_from_slice(&if self.cmaf { ftyp_cmaf() } else { ftyp() });
            let track = TrackInit::Video {
                codec: self.codec,
                width: self.width,
                height: self.height,
                param_sets: param_sets.iter().map(|set| set.to_vec()).collect(),
            };
            let metadata = MoovMetadata {
                global: self.tags.clone(),
                per_track: Vec::new(),
                chapters: self.chapters.clone(),
            };
            out.extend_from_slice(&av_moov(&[Some(track)], None, &metadata, &[]));
            self.header_written = true;
        }
        // A zero `dts_ns` means upstream is not timing decode order, except on
        // the first sample, where it is also where a reordered stream starts. A
        // DTS past the PTS is not expressible in a version 0 `trun`.
        let first_sample = self.prev_decode_ns.is_none();
        let decode_ns = match dts_ns {
            d if d <= pts_ns && (d > 0 || first_sample) => d,
            _ => pts_ns,
        };
        // duration: explicit, else decode-order delta, else one caps frame.
        let duration_ns = if duration_ns != 0 {
            duration_ns
        } else {
            match self.prev_decode_ns {
                Some(prev) if decode_ns > prev => decode_ns - prev,
                _ => self.frame_ns,
            }
        };
        self.prev_decode_ns = Some(decode_ns);
        let duration = ns_to_ticks(duration_ns, VIDEO_TIMESCALE) as u32;
        let composition_offset = ns_to_ticks(pts_ns - decode_ns, VIDEO_TIMESCALE) as u32;
        let sample = avcc_sample(&nalus);
        let is_sync = au_is_keyframe(self.codec, annexb);

        // Default: one access unit per fragment. CMAF mode always batches instead,
        // because a fragment there may only start at a sync sample.
        if self.fragment_duration_ns == 0 && !self.cmaf {
            if let Some(now) = self.producer_clock {
                out.extend_from_slice(&prft(TRACK_ID, now(), self.decode_time));
            }
            let fragment_sample = FragmentSample {
                bytes: &sample,
                duration,
                is_sync,
                composition_offset,
            };
            av_fragment(
                &mut out,
                TRACK_ID,
                self.sequence + 1,
                self.decode_time,
                false,
                &[fragment_sample],
            );
            self.sequence += 1;
            self.decode_time += duration as u64;
            return Ok(out);
        }

        // Batched: close the open fragment at a sync sample once it has reached the
        // target duration, so each fragment begins at a keyframe. With chunking on,
        // the open chunk is only part of it, so the target is measured against the
        // whole fragment.
        let open = self.fragment_started || !self.pending.is_empty();
        if is_sync && open && self.fragment_dur_ns >= self.fragment_duration_ns {
            out.extend_from_slice(&self.flush_pending());
            self.fragment_started = false;
            self.fragment_dur_ns = 0;
        }
        if self.pending.is_empty() {
            self.pending_decode_time = self.decode_time;
        }
        if !self.fragment_started && self.pending.is_empty() {
            // the producer's wall clock at the fragment's first sample is what its
            // prft maps to that sample's decode time.
            self.fragment_ntp = self.producer_clock.map_or(0, |now| now());
        }
        self.pending.push(PendingSample {
            bytes: sample,
            duration,
            is_sync,
            composition_offset,
        });
        self.pending_dur_ns = self.pending_dur_ns.saturating_add(duration_ns);
        self.fragment_dur_ns = self.fragment_dur_ns.saturating_add(duration_ns);
        // Close the chunk as soon as it reaches its target: the point of chunking
        // is that these bytes leave now rather than at the end of the fragment.
        if self.chunk_duration_ns > 0 && self.pending_dur_ns >= self.chunk_duration_ns {
            out.extend_from_slice(&self.flush_pending());
        }
        Ok(out)
    }

    /// Emit the open chunk (the whole open fragment when chunking is off),
    /// advancing the decode clock. The chunk that opens a fragment carries the
    /// fragment's `styp` and `prft`; the rest continue it. Empty when nothing is
    /// buffered.
    fn flush_pending(&mut self) -> Vec<u8> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let samples: Vec<FragmentSample<'_>> = self
            .pending
            .iter()
            .map(PendingSample::as_fragment_sample)
            .collect();
        let mut frag = Vec::new();
        if !self.fragment_started {
            if self.cmaf {
                // this fragment is a CMAF segment, of chunks when chunking is on
                frag.extend_from_slice(&styp_cmaf(self.chunk_duration_ns > 0));
            }
            if self.producer_clock.is_some() {
                frag.extend_from_slice(&prft(
                    TRACK_ID,
                    self.fragment_ntp,
                    self.pending_decode_time,
                ));
            }
        }
        av_fragment(
            &mut frag,
            TRACK_ID,
            self.sequence + 1,
            self.pending_decode_time,
            self.cmaf,
            &samples,
        );
        let total: u64 = self.pending.iter().map(|s| s.duration as u64).sum();
        self.sequence += 1;
        self.decode_time += total;
        self.pending.clear();
        self.pending_dur_ns = 0;
        self.fragment_started = true;
        frag
    }

    /// Flush the final partial fragment at end of stream (batched mode). A no-op
    /// in per-AU mode (nothing is ever buffered).
    pub(crate) fn flush(&mut self) -> Vec<u8> {
        let tail = self.flush_pending();
        self.fragment_started = false;
        self.fragment_dur_ns = 0;
        tail
    }
}

// The Annex-B split / NAL-type / parameter-set / AVCC helpers moved to the
// ungated `annexb` module (M662, the no_std FLV muxer shares them);
// re-exported so this module's users keep their import path.
pub(crate) use crate::annexb::{avcc_record, avcc_sample, parameter_sets, split_annexb};

// box primitives (mp4_box/full_box/ftyp) shared across the MP4 elements.
use crate::mp4box::{ftyp, ftyp_cmaf, mp4_box, prft, styp_cmaf};

/// The one track's `track_ID`.
const TRACK_ID: u32 = 1;

// --- box writers ----------------------------------------------------------

/// The visual sample entry for the track: `avc1`+`avcC` for H.264,
/// `hvc1`+`hvcC` for H.265. `param_sets` is the ordered set the moov needs
/// (SPS,PPS for H.264; VPS,SPS,PPS for H.265).
pub(crate) fn visual_sample_entry(
    codec: VideoCodec,
    width: u32,
    height: u32,
    param_sets: &[&[u8]],
) -> Vec<u8> {
    let (fourcc, config): (&[u8; 4], Vec<u8>) = match codec {
        VideoCodec::H265 => (b"hvc1", hvcc(param_sets)),
        _ => (b"avc1", avcc(param_sets)),
    };

    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 16]); // pre_defined/reserved
    p.extend_from_slice(&(width as u16).to_be_bytes());
    p.extend_from_slice(&(height as u16).to_be_bytes());
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    p.extend_from_slice(&[0u8; 4]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // frame count
    p.extend_from_slice(&[0u8; 32]); // compressor name
    p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
    p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    p.extend_from_slice(&config);
    mp4_box(fourcc, &p)
}

/// `avcC` decoder configuration record box. `param_sets` is [SPS, PPS].
fn avcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"avcC", &avcc_record(param_sets))
}

/// One Annex-B video access unit as the MP4 and Matroska muxers store it, and
/// whether it is a sync sample: H.26x as AVCC length-prefixed NALUs, AV1 without
/// temporal delimiters, VP8 / VP9 verbatim.
pub(crate) fn video_sample(codec: VideoCodec, au: &[u8]) -> (Vec<u8>, bool) {
    match codec {
        VideoCodec::H264 | VideoCodec::H265 => {
            (avcc_sample(&split_annexb(au)), au_is_keyframe(codec, au))
        }
        VideoCodec::Vp8 => (au.to_vec(), vp8_keyframe(au)),
        VideoCodec::Av1 => (
            crate::av1parse::strip_temporal_delimiters(au),
            crate::av1parse::av1_keyframe(au),
        ),
        _ => (au.to_vec(), vp9_keyframe(au)),
    }
}

/// VP8 keyframe flag: the frame tag's bit 0 (`0` = key frame). Shared by the
/// container muxers that store VP8 frames verbatim (Matroska, MP4).
pub(crate) fn vp8_keyframe(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// VP9 keyframe flag from the uncompressed frame header: frame_marker(2)=0b10,
/// profile(2) (+1 reserved bit for profile 3), show_existing_frame(1), then
/// frame_type(1) where `0` = key frame. Superframes are not unpacked (the vpx
/// encoder emits a single frame per buffer). Shared by the VP9-carrying muxers.
pub(crate) fn vp9_keyframe(frame: &[u8]) -> bool {
    let Some(&b0) = frame.first() else {
        return false;
    };
    let bit = |i: u32| (b0 >> (7 - i)) & 1;
    if ((bit(0) << 1) | bit(1)) != 0b10 {
        return false; // not a valid VP9 frame marker
    }
    let profile = (bit(3) << 1) | bit(2);
    let mut cursor: u32 = 4;
    if profile == 3 {
        cursor += 1; // reserved_zero
    }
    if bit(cursor) == 1 {
        return false; // show_existing_frame: a repeat, not a key frame
    }
    cursor += 1;
    bit(cursor) == 0 // frame_type: 0 = key frame
}

/// `hvcC` decoder configuration record. `param_sets` is [VPS, SPS, PPS]. The
/// 12-byte general profile_tier_level is copied from the SPS (it sits right
/// after the 2-byte NAL header and the 1-byte sps_video_parameter_set_id /
/// max_sub_layers / nesting field); the remaining descriptive fields are set
/// to the 4:2:0 8-bit defaults the MS HEVC encoder produces. Parameter sets
/// stay in-band in each sample regardless, so a player re-parses authoritative
/// values from the SPS.
fn hvcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"hvcC", &hvcc_record(param_sets))
}

/// The `hvcC` HEVCDecoderConfigurationRecord body (no box header), shared as the
/// Matroska `CodecPrivate` for `V_MPEGH/ISO/HEVC`. See [`hvcc`] for the field
/// layout; `param_sets` is [VPS, SPS, PPS].
pub(crate) fn hvcc_record(param_sets: &[&[u8]]) -> Vec<u8> {
    let vps = param_sets[0];
    let sps = param_sets[1];
    let pps = param_sets[2];

    let mut ptl = [0u8; 12];
    if let Some(src) = sps.get(3..15) {
        ptl.copy_from_slice(src);
    }

    let mut p = Vec::new();
    p.push(1); // configuration version
    p.extend_from_slice(&ptl); // general profile_tier_level (12 bytes)
    p.extend_from_slice(&0xF000u16.to_be_bytes()); // reserved + min_spatial_segmentation_idc 0
    p.push(0xFC); // reserved + parallelismType 0
    p.push(0xFC | 1); // reserved + chromaFormat 4:2:0
    p.push(0xF8); // reserved + bitDepthLumaMinus8 0
    p.push(0xF8); // reserved + bitDepthChromaMinus8 0
    p.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate (unspecified)
                                              // constantFrameRate(2)=0 | numTemporalLayers(3)=1 | temporalIdNested(1)=1
                                              // | lengthSizeMinusOne(2)=3
    p.push((1 << 3) | (1 << 2) | 3);
    p.push(3); // numOfArrays: VPS, SPS, PPS

    for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
        p.push(0x80 | ty); // array_completeness=1, NAL_unit_type
        p.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
        p.extend_from_slice(nalu);
    }
    p
}

/// The `AV1CodecConfigurationRecord` body (no box header): the `av1C` box
/// payload and the Matroska `CodecPrivate` for `V_AV1`. Header fields come
/// from the parsed sequence header; `seq_obu` (the whole sequence-header OBU)
/// rides as `configOBUs` verbatim, so a player re-parses authoritative values.
pub(crate) fn av1c_record(seq: &crate::av1parse::Av1SeqHeader, seq_obu: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + seq_obu.len());
    p.push(0x81); // marker(1)=1 | version(7)=1
    p.push((seq.profile << 5) | (seq.level & 0x1F));
    p.push(
        ((seq.tier & 1) << 7)
            | ((seq.high_bitdepth as u8) << 6)
            | ((seq.twelve_bit as u8) << 5)
            | ((seq.monochrome as u8) << 4)
            | ((seq.subsampling_x as u8) << 3)
            | ((seq.subsampling_y as u8) << 2)
            | (seq.chroma_sample_position & 3),
    );
    p.push(0); // reserved + no initial_presentation_delay
    p.extend_from_slice(seq_obu);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::nalu_type;
    use alloc::vec;

    #[test]
    fn annexb_splitter_handles_both_start_codes() {
        // 4-byte code, 3-byte code, trailing NALU
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, // SPS
            0, 0, 1, 0x68, 0xBB, // PPS
            0, 0, 0, 1, 0x65, 0xCC, 0xDD, // IDR
        ];
        let nalus = split_annexb(&data);
        assert_eq!(nalus.len(), 3);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[0]), 7);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[1]), 8);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[2]), 5);
        assert_eq!(nalus[2], &[0x65, 0xCC, 0xDD]);
    }

    #[test]
    fn hevc_nal_type_reads_header_bits() {
        // H.265 NAL type is bits 1..6 of byte 0. VPS=32 -> 0x40, IDR_W_RADL=19
        // -> 0x26.
        assert_eq!(nalu_type(VideoCodec::H265, &[0x40, 0x01]), 32);
        assert_eq!(nalu_type(VideoCodec::H265, &[0x26, 0x01]), 19);
    }

    #[test]
    fn hvcc_carries_three_arrays_and_copies_ptl() {
        // synthetic VPS(32)/SPS(33)/PPS(34); SPS holds a 12-byte PTL after its
        // 3-byte prefix so hvcC copies it verbatim.
        let vps: &[u8] = &[0x40, 0x01, 0xAA];
        let mut sps_v = vec![0x42u8, 0x01, 0x00];
        sps_v.extend((1u8..=12).map(|b| b * 7)); // recognisable PTL bytes
        let sps: &[u8] = &sps_v;
        let pps: &[u8] = &[0x44, 0x01, 0xCC];
        let cfg = hvcc(&[vps, sps, pps]);
        assert_eq!(&cfg[4..8], b"hvcC");
        let payload = &cfg[8..];
        assert_eq!(payload[0], 1, "configuration version");
        assert_eq!(
            &payload[1..13],
            &sps_v[3..15],
            "general PTL copied from SPS"
        );
        assert_eq!(payload[22], 3, "numOfArrays = VPS, SPS, PPS");
        // the three parameter sets must appear in the record
        assert!(cfg.windows(vps.len()).any(|w| w == vps));
        assert!(cfg.windows(pps.len()).any(|w| w == pps));
    }

    #[test]
    fn avcc_sample_length_prefixes_every_nalu() {
        let nalus: Vec<&[u8]> = vec![&[0x67, 1, 2], &[0x65, 3]];
        let s = avcc_sample(&nalus);
        assert_eq!(
            s,
            vec![0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3],
            "4-byte BE length before each NALU"
        );
    }

    #[test]
    fn avcc_sample_drops_empty_nalus() {
        let nalus: Vec<&[u8]> = vec![&[0x67, 1], &[], &[0x65, 3]];
        assert_eq!(
            avcc_sample(&nalus),
            vec![0, 0, 0, 2, 0x67, 1, 0, 0, 0, 2, 0x65, 3]
        );
    }

    #[test]
    fn boxes_carry_size_and_type() {
        let b = mp4_box(b"mdat", &[1, 2, 3]);
        assert_eq!(&b[..4], &11u32.to_be_bytes());
        assert_eq!(&b[4..8], b"mdat");
        assert_eq!(&b[8..], &[1, 2, 3]);
    }

    /// Chunking splits one fragment into several `moof`+`mdat` pairs, and only
    /// the chunk that opens the fragment carries its `styp` and the `prft` built
    /// from the injected producer clock.
    #[test]
    fn chunks_carry_one_styp_and_one_prft_per_fragment() {
        const NTP: u64 = 0x1122_3344_5566_7788;
        let au = |nals: &[&[u8]]| {
            let mut v = Vec::new();
            for n in nals {
                v.extend_from_slice(&[0, 0, 0, 1]);
                v.extend_from_slice(n);
            }
            v
        };
        let key = au(&[
            &[0x67, 0x42, 0x00, 0x1e, 0x88],
            &[0x68, 0xce, 0x3c, 0x80],
            &[0x65, 0x88, 0x84, 0x00],
        ]);
        let inter = au(&[&[0x41, 0x9a, 0x00]]);

        // Four 40 ms samples chunked at 80 ms: two chunks of two, one fragment.
        let mut m = Fmp4Muxer::new(VideoCodec::H264, 320, 240, TagList::new())
            .with_cmaf(true)
            .with_chunk_duration_ns(80_000_000)
            .with_producer_clock(Some(|| NTP));
        let mut out = Vec::new();
        for i in 0..4u64 {
            let data = if i == 0 { &key } else { &inter };
            out.extend_from_slice(&m.push_au(data, i * 40_000_000, 0, 40_000_000).unwrap());
        }
        out.extend_from_slice(&m.flush());

        let count = |four: &[u8; 4]| out.windows(4).filter(|w| *w == four).count();
        assert_eq!(count(b"moof"), 2, "one moof per chunk");
        assert_eq!(count(b"styp"), 1, "one segment, not one per chunk");
        assert_eq!(
            count(b"prft"),
            1,
            "one producer reference time per fragment"
        );

        // styp, then prft, then the moof they introduce.
        let at = |four: &[u8; 4]| out.windows(4).position(|w| w == four).unwrap();
        assert!(at(b"styp") < at(b"prft") && at(b"prft") < at(b"moof"));

        // prft payload: version 1, reference track 1, the injected clock, and the
        // fragment's own base decode time (its first chunk starts the timeline).
        let p = at(b"prft") + 4;
        assert_eq!(out[p], 1, "version 1 (64-bit media_time)");
        assert_eq!(&out[p + 1..p + 4], &[0, 0, 0], "flags 0: an encoder time");
        assert_eq!(&out[p + 4..p + 8], &1u32.to_be_bytes(), "reference track");
        assert_eq!(&out[p + 8..p + 16], &NTP.to_be_bytes());
        assert_eq!(&out[p + 16..p + 24], &0u64.to_be_bytes(), "media time");
    }

    /// The default per-access-unit mode takes the early path in `push_au`, so
    /// it writes the `prft` there: one per fragment, i.e. one per sample.
    #[test]
    fn per_au_fragments_carry_a_prft_each() {
        const NTP: u64 = 0x0102_0304_0506_0708;
        let mut key = Vec::new();
        for n in [
            &[0x67, 0x42, 0x00, 0x1e, 0x88][..],
            &[0x68, 0xce, 0x3c, 0x80],
            &[0x65, 0x88, 0x84, 0x00],
        ] {
            key.extend_from_slice(&[0, 0, 0, 1]);
            key.extend_from_slice(n);
        }
        let mut m = Fmp4Muxer::new(VideoCodec::H264, 320, 240, TagList::new())
            .with_producer_clock(Some(|| NTP));
        let mut out = Vec::new();
        for i in 0..3u64 {
            out.extend_from_slice(&m.push_au(&key, i * 40_000_000, 0, 40_000_000).unwrap());
        }
        out.extend_from_slice(&m.flush());

        let count = |four: &[u8; 4]| out.windows(4).filter(|w| *w == four).count();
        assert_eq!(count(b"moof"), 3, "one fragment per access unit");
        assert_eq!(count(b"prft"), 3, "a prft ahead of every fragment");
        let p = out.windows(4).position(|w| w == b"prft").unwrap() + 4;
        assert_eq!(&out[p + 8..p + 16], &NTP.to_be_bytes());
    }
    /// An encoder's frames carry no duration. The first sample then lasts one
    /// caps frame (not the 30 fps default), the rest the decode-order delta, and
    /// a reordered stream's `trun` carries `pts - dts` per sample.
    #[test]
    fn untimed_samples_take_the_caps_frame_period_and_keep_reorder_offsets() {
        use crate::fmp4::parse_trun;
        use crate::mp4box::find_box;
        const FRAME_NS: u64 = 40_000_000;
        let au = |nals: &[&[u8]]| {
            let mut v = Vec::new();
            for n in nals {
                v.extend_from_slice(&[0, 0, 0, 1]);
                v.extend_from_slice(n);
            }
            v
        };
        let key = au(&[
            &[0x67, 0x42, 0x00, 0x1e, 0x88],
            &[0x68, 0xce, 0x3c, 0x80],
            &[0x65, 0x88, 0x84, 0x00],
        ]);
        let inter = au(&[&[0x41, 0x9a, 0x00]]);
        let mut m = Fmp4Muxer::new(VideoCodec::H264, 320, 240, TagList::new())
            .with_frame_duration_ns(FRAME_NS);
        // decode order I P B: (pts, dts) with a one-frame reorder delay
        let frames = [
            (&key, FRAME_NS, 0),
            (&inter, 3 * FRAME_NS, FRAME_NS),
            (&inter, 2 * FRAME_NS, 2 * FRAME_NS),
        ];
        let mut got = Vec::new();
        for (data, pts, dts) in frames {
            let out = m.push_au(data, pts, dts, 0).unwrap();
            let moof_at = out.windows(4).position(|w| w == b"moof").unwrap() - 4;
            let moof_len = u32::from_be_bytes(out[moof_at..moof_at + 4].try_into().unwrap());
            let moof = &out[moof_at + 8..moof_at + moof_len as usize];
            let traf = find_box(moof, b"traf").unwrap();
            let trun = find_box(traf, b"trun").unwrap();
            let samples = parse_trun(trun, 0, 0, out.len()).unwrap();
            assert_eq!(samples.len(), 1);
            got.push((samples[0].duration, samples[0].composition_offset));
        }
        let tick = |ns: u64| ns_to_ticks(ns, VIDEO_TIMESCALE);
        assert_eq!(
            got,
            [
                (tick(FRAME_NS) as u32, tick(FRAME_NS) as i64),
                (tick(FRAME_NS) as u32, tick(2 * FRAME_NS) as i64),
                (tick(FRAME_NS) as u32, 0),
            ],
            "(duration, composition offset) per sample"
        );
    }
}

//! Fragmented-MP4 / CMAF parsing shared by [`Mp4Src`](crate::mp4src) (file
//! source) and [`Fmp4Demux`](crate::fmp4demux) (byte-stream demuxer). Pure
//! `no_std + alloc`: reads the `moov` init (codec, geometry, timescale,
//! parameter sets) and walks `moof`+`mdat` fragments into Annex-B samples.
//!
//! Supported profile: one video track, `trun` v0 with explicit sample sizes,
//! `default-base-is-moof` data offsets landing on the following `mdat`'s
//! payload (what `mp4mux` writes and CMAF single-track files share). Anything
//! else fails loud rather than emitting a corrupt bitstream.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::log::Target;
use g2g_core::meta::{CropRectangle, FrameMetaSet};
use g2g_core::time::ticks_to_ns;

use g2g_core::{
    g2g_warn, AudioFormat, Chapter, ClosedCaptionFormat, FrameTiming, G2gError, ProResVariant,
    RawVideoFormat, Segment, TagList, TextFormat, VideoCodec,
};

use crate::aiff::audio_from_comm;
use crate::annexb::{au_is_keyframe, intra_only};
use crate::avidemux::is_pcm;
use crate::cea::{parse_cdp, write_cc_data, CcTriple};
use crate::cenc::{
    fragment_sample_crypt, parse_movie_seig, parse_sinf, CencDefaults, CencTrack, SampleCrypt,
};
#[cfg(any(test, feature = "dash"))]
use crate::mp4box::next_box_len;
use crate::mp4box::{
    be32, be64, boxes, boxes_at, find_box, find_path, itunes_gapless, parse_chpl, parse_esds,
    parse_esds_decoder_config, parse_ilst_tags,
};
use crate::opusparse::{opus_head_from_dops, parse_opus_head};
use crate::pcmclip::clip_pcm_to_segment;
use crate::pcmendian::convert_wire_layout;
use crate::uncompressedvideo::{copy_raw_video_frame, quicktime_raw_video, RawVideoStorage};

#[derive(Debug)]
pub(crate) struct Header {
    pub(crate) codec: VideoCodec,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) timescale: u32,
    /// Total track duration in nanoseconds from `mdhd`, or `None` when the box
    /// reports `0` (a fragmented / live init segment whose length is unknown
    /// until the fragments arrive). Feeds the M203 `DURATION` query.
    pub(crate) duration_ns: Option<u64>,
    // None when the stts deltas vary or the table is empty (a fragmented init)
    pub(crate) framerate_q16: Option<u32>,
    /// Parameter-set NALUs in container order (SPS,PPS for H.264; VPS,SPS,PPS
    /// for H.265), prepended to the first sample if it carries none in-band.
    pub(crate) param_sets: Vec<Vec<u8>>,
    /// Common-encryption metadata from the init's `tenc` (plus any movie-level
    /// `seig` table), `None` for a clear track.
    pub(crate) cenc: Option<CencTrack>,
    pub(crate) clean_aperture: Option<CropRectangle>,
}

/// In-place sample decryptor: given the crypto resolved for a sample (scheme, IV,
/// pattern, KID, subsample map), the byte offset of its fragment in the source
/// stream, and the sample's bytes, rewrites the protected ranges. The demuxers
/// supply one that resolves the content key from the KID and that offset (a
/// rotating `#EXT-X-KEY` changes key by stream position); a sample whose key is
/// unavailable fails the parse rather than emitting garbage.
pub(crate) type SampleDecrypt<'a> =
    &'a mut dyn FnMut(&SampleCrypt, u64, &mut [u8]) -> Result<(), G2gError>;

#[derive(Debug)]
pub(crate) struct Sample {
    pub(crate) annexb: Vec<u8>,
    pub(crate) pts_ns: u64,
    /// When the sample decodes, which a reordered (B-frame) stream places ahead
    /// of its `pts_ns`.
    pub(crate) dts_ns: u64,
    pub(crate) duration_ns: u64,
    /// Whether the access unit carries an IDR picture (a seek snap point).
    pub(crate) keyframe: bool,
}

/// The index of the keyframe at or before `target_ns` (GStreamer `SNAP_BEFORE`,
/// so a decoder can resume from a clean reference); 0 when none precedes it.
pub(crate) fn keyframe_index_for(samples: &[Sample], target_ns: u64) -> usize {
    samples
        .iter()
        .enumerate()
        .rfind(|(_, s)| s.keyframe && s.pts_ns <= target_ns)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// seek targets and positions count from the earliest pts, as ffmpeg -ss counts from the start time
pub(crate) fn stream_start_ns(samples: &[Sample]) -> u64 {
    samples
        .iter()
        .map(|sample| sample.pts_ns)
        .min()
        .unwrap_or(0)
}

pub(crate) fn every_sample_ends_by(samples: &[Sample], target_ns: u64) -> bool {
    samples
        .iter()
        .all(|s| s.pts_ns.saturating_add(s.duration_ns) <= target_ns)
}

/// Parse the `moov` init box into a [`Header`] (codec, geometry, timescale,
/// parameter sets). `data` must contain the `moov` (a whole init segment or a
/// whole file).
pub(crate) fn parse_header(data: &[u8]) -> Result<Header, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;

    let tkhd = read_tkhd(find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?)?;
    let (width, height) = (tkhd.width, tkhd.height);

    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let mdhd = read_mdhd(find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?)?;
    let timescale = mdhd.timescale;
    // `0` means the length is not yet known (a fragmented init segment).
    let duration_ns = match mdhd.duration {
        0 => None,
        units => Some((units as u128 * 1_000_000_000 / timescale as u128) as u64),
    };

    // stsd's first entry is the visual sample entry: avc1/avcC (H.264) or
    // hvc1/hev1 with hvcC (H.265). Its config record carries the parameter sets.
    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    let stsd = find_box(stbl, b"stsd").ok_or(G2gError::CapsMismatch)?;
    // full box: version/flags + entry count, then the first sample entry.
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
    // visual sample entry: 78 bytes of fixed fields before the nested boxes. An
    // encrypted track uses `encv`, carrying the original codec config plus a
    // `sinf` (frma original format + cbcs scheme + tenc defaults).
    let (codec, param_sets, cenc) = if let Some(avc1) = find_box(entries, b"avc1") {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H264, parse_avcc(avcc)?.parameter_sets, None)
    } else if let Some(hvc1) = find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1")) {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?.parameter_sets, None)
    } else if let Some(mp4v) = find_box(entries, b"mp4v") {
        let (codec, sets) = mp4v_codec(mp4v)?.ok_or(G2gError::CapsMismatch)?;
        (codec, sets, None)
    } else if let Some(av01) = find_box(entries, b"av01") {
        (VideoCodec::Av1, parse_av1c_config(av01)?, None)
    } else if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_sinf(sinf)?;
        let frma = find_box(sinf, b"frma").ok_or(G2gError::CapsMismatch)?;
        let (codec, param_sets) = match frma.get(0..4) {
            Some(b"avc1") => {
                let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H264, parse_avcc(avcc)?.parameter_sets)
            }
            Some(b"hvc1") | Some(b"hev1") => {
                let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H265, parse_hvcc(hvcc)?.parameter_sets)
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        (codec, param_sets, Some(cenc))
    } else {
        return Err(G2gError::CapsMismatch);
    };

    Ok(Header {
        codec,
        width,
        height,
        timescale,
        duration_ns,
        framerate_q16: constant_framerate_q16(stbl, timescale),
        param_sets,
        cenc: cenc_track(cenc, stbl)?,
        clean_aperture: clean_aperture(entries),
    })
}

#[derive(Debug)]
struct TrackHeaderBox {
    track_id: u32,
    width: u32,
    height: u32,
}

// a version 1 tkhd / mdhd widens the creation and modification times to 64 bits
const VERSION_1_TIME_WIDENING: usize = 8;
const VERSION_1_DURATION_WIDENING: usize = 4;

fn read_tkhd(tkhd: &[u8]) -> Result<TrackHeaderBox, G2gError> {
    let (track_id_at, width_at) = match tkhd.first() {
        Some(0) => (12, 76),
        Some(1) => (
            12 + VERSION_1_TIME_WIDENING,
            76 + VERSION_1_TIME_WIDENING + VERSION_1_DURATION_WIDENING,
        ),
        _ => return Err(G2gError::CapsMismatch),
    };
    Ok(TrackHeaderBox {
        track_id: be32(tkhd, track_id_at)?,
        width: be32(tkhd, width_at)? >> 16,
        height: be32(tkhd, width_at + 4)? >> 16,
    })
}

#[derive(Debug)]
struct MediaHeaderBox {
    timescale: u32,
    duration: u64,
}

fn read_mdhd(mdhd: &[u8]) -> Result<MediaHeaderBox, G2gError> {
    let (timescale, duration) = match mdhd.first() {
        Some(0) => (be32(mdhd, 12)?, u64::from(be32(mdhd, 16)?)),
        Some(1) => (
            be32(mdhd, 12 + VERSION_1_TIME_WIDENING)?,
            be64(mdhd, 16 + VERSION_1_TIME_WIDENING)?,
        ),
        _ => return Err(G2gError::CapsMismatch),
    };
    if timescale == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(MediaHeaderBox {
        timescale,
        duration,
    })
}

// stts: a 4-byte version/flags and a 4-byte entry count, then (count, delta) pairs
const STTS_FIRST_ENTRY: usize = 8;
const STTS_ENTRY_LEN: usize = 8;
const STTS_DELTA_OFFSET: usize = 4;

// timescale / delta when every sample in the stts lasts the same delta
fn constant_framerate_q16(stbl: &[u8], timescale: u32) -> Option<u32> {
    let stts = find_box(stbl, b"stts")?;
    let entries = be32(stts, 4).ok()?;
    let mut constant_delta = None;
    for entry in 0..entries as usize {
        let offset = entry
            .checked_mul(STTS_ENTRY_LEN)?
            .checked_add(STTS_FIRST_ENTRY)?;
        let count = be32(stts, offset).ok()?;
        let delta = be32(stts, offset.checked_add(STTS_DELTA_OFFSET)?).ok()?;
        if count == 0 {
            continue;
        }
        if *constant_delta.get_or_insert(delta) != delta {
            return None;
        }
    }
    framerate_q16_from_fraction(timescale, constant_delta?).filter(|&q16| q16 > 0)
}

// a fragmented init leaves stts empty, so the fragment's tfhd declares the sample duration
pub(crate) fn fragment_framerate_q16(moof: &[u8], timescale: u32) -> Option<u32> {
    let tfhd = find_path(moof, &[b"traf", b"tfhd"])?;
    let defaults = tfhd_defaults(tfhd).ok()?;
    framerate_q16_from_fraction(timescale, defaults.duration).filter(|&q16| q16 > 0)
}

// tfhd payload: version/flags, then track_ID
const TFHD_TRACK_ID_AT: usize = 4;

// the constant frame rate of one track of a whole file, from its stts or its first fragment's tfhd
pub(crate) fn track_framerate_q16(data: &[u8], track: &TrackHeader) -> Option<u32> {
    let moov = find_box(data, b"moov")?;
    let stbl = find_path(
        find_trak_by_id(moov, track.track_id)?,
        &[b"mdia", b"minf", b"stbl"],
    )?;
    constant_framerate_q16(stbl, track.timescale)
        .or_else(|| track_fragment_framerate_q16(data, track))
}

// from the tfhd of the first fragment in `data`, else the moov's trex
pub(crate) fn track_fragment_framerate_q16(data: &[u8], track: &TrackHeader) -> Option<u32> {
    let tfhd_duration = find_box(data, b"moof")
        .and_then(|moof| {
            boxes(moof)
                .filter(|(kind, _)| *kind == b"traf")
                .filter_map(|(_, traf)| find_box(traf, b"tfhd"))
                .find(|tfhd| be32(tfhd, TFHD_TRACK_ID_AT).ok() == Some(track.track_id))
        })
        .and_then(|tfhd| tfhd_defaults(tfhd).ok())
        .map_or(0, |defaults| defaults.duration);
    let duration = if tfhd_duration == 0 {
        track.trex_defaults.duration
    } else {
        tfhd_duration
    };
    framerate_q16_from_fraction(track.timescale, duration).filter(|&q16| q16 > 0)
}

/// Pair a track's `tenc` defaults with the movie-level `seig` table its `stbl`
/// carries, which a fragment's `sbgp` can index instead of a `traf`-local one.
fn cenc_track(defaults: Option<CencDefaults>, stbl: &[u8]) -> Result<Option<CencTrack>, G2gError> {
    match defaults {
        Some(defaults) => {
            let movie_seig = parse_movie_seig(stbl, defaults.scheme)?;
            Ok(Some(CencTrack {
                defaults,
                movie_seig,
            }))
        }
        None => Ok(None),
    }
}

/// What one track carries: a video elementary stream (codec + geometry +
/// parameter sets), an audio elementary stream (format + channel layout + codec
/// config), or a timed-text stream (subtitle format). The multi-track read-side
/// analog of `TrackInit` in the muxer; clear tracks only (encryption stays
/// single-track via [`parse_header`]).
#[derive(Debug, Clone)]
pub(crate) enum TrackKind {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
        param_sets: Vec<Vec<u8>>,
    },
    // geometry from the sample entry, which the stored frames follow whatever tkhd displays
    RawVideo {
        format: RawVideoFormat,
        width: u32,
        height: u32,
        storage: RawVideoStorage,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
        /// The track's out-of-band codec config, in the form the elementary
        /// stream carries it: AAC's AudioSpecificConfig (`esds`), or an Opus
        /// `OpusHead` rebuilt from the `dOps` (M791). Empty when there is none.
        config: Vec<u8>,
        /// How a PCM track stores its samples, `None` for a coded format.
        stored_pcm: Option<StoredPcm>,
    },
    /// A timed-text subtitle track. The container carries the per-cue timing
    /// (sample PTS + duration); `format` names the elementary cue payload's syntax
    /// downstream, and `sample` the on-disk sample-entry framing (which selects
    /// de-framing): `tx3g` / `wvtt` cues are plain UTF-8 ([`TextFormat::Utf8`]),
    /// `stpp` samples are TTML documents ([`TextFormat::Ttml`]).
    Text {
        format: TextFormat,
        sample: TextSampleFormat,
    },
    /// A raw closed-caption track (`c608` / `c708`): the caption data is its own
    /// elementary stream rather than SEI inside the video. Samples de-frame from
    /// their container atoms to the `cc_data` triple stream
    /// [`Caps::ClosedCaption`](g2g_core::Caps::ClosedCaption) carries.
    ClosedCaption { format: ClosedCaptionFormat },
}

/// How a QuickTime PCM track's stored samples differ from its graph format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoredPcm {
    /// AIFF's layout: 8-bit samples signed, wider ones in `big_endian` order.
    Aiff { big_endian: bool },
    /// `raw `: unsigned 8-bit, already the graph layout.
    Unsigned8,
}

impl StoredPcm {
    fn to_graph(self, samples: &mut [u8], format: AudioFormat) {
        if let StoredPcm::Aiff { big_endian } = self {
            convert_wire_layout(samples, format, big_endian);
        }
    }
}

/// The on-disk timed-text sample format an MP4 text `trak` stores. Distinct from
/// the elementary [`TextFormat`] the de-framed cue carries: both `tx3g` and `wvtt`
/// de-frame to plain UTF-8, but by different framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextSampleFormat {
    /// 3GPP / QuickTime timed text: 2-byte length prefix + UTF-8 cue + style boxes.
    Tx3g,
    /// WebVTT-in-MP4 (ISO 14496-30): per-sample `vttc` / `vtte` boxes; the `payl`
    /// payloads concatenate to the cue's UTF-8 text.
    Wvtt,
    /// TTML-in-MP4 (`stpp`): each sample is a complete XML document, passed through.
    Stpp,
}

/// One track's init data parsed from a `moov/trak`: the `track_ID` (which keys
/// the fragments in [`parse_fragments_multi`]), the media timescale, the
/// elementary-stream kind, the common-encryption metadata for an encrypted track
/// (`None` for a clear one), and the track's own `udta/meta/ilst` metadata
/// (empty when it carries none).
#[derive(Debug, Clone)]
pub(crate) struct TrackHeader {
    pub(crate) track_id: u32,
    pub(crate) timescale: u32,
    pub(crate) kind: TrackKind,
    pub(crate) cenc: Option<CencTrack>,
    pub(crate) tags: TagList,
    /// Where the presentation starts in the track's media timeline (timescale
    /// ticks): the `edts/elst` first edit's `media_time`, e.g. an AAC encoder's
    /// 1024-sample priming. `0` when there is no edit list or it has no media
    /// edit.
    pub(crate) media_start: u64,
    // timescale ticks, exclusive: the end of the elst's last edit
    pub(crate) media_stop: Option<u64>,
    // the media between consecutive edits that neither presents, in order
    pub(crate) edit_gaps: Vec<EditGap>,
    // a video edit list that replays or reorders media, laid out one edit after another
    pub(crate) replayed_edits: Vec<ReplayedEdit>,
    /// The `mvex/trex` default sample duration, size and flags, which a
    /// fragment falls back to when its `tfhd` does not declare them.
    pub(crate) trex_defaults: SampleDefaults,
    pub(crate) clean_aperture: Option<CropRectangle>,
}

// media ticks: where the edit starts presenting, counted from the first edit's start, and the media it presents
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReplayedEdit {
    presented_from: u64,
    media: Range<u64>,
}

// media time between two edits, in ns
#[derive(Debug, Clone)]
pub(crate) struct EditGap {
    media_ns: Range<u64>,
    // the media all the gaps before this one cut
    earlier_gaps_ns: u64,
}

#[cfg(feature = "metadata")]
pub(crate) fn clean_aperture_meta(clean_aperture: Option<CropRectangle>) -> FrameMetaSet {
    let mut meta = FrameMetaSet::new();
    if let Some(rectangle) = clean_aperture {
        meta.attach(g2g_core::meta::CropMeta { rectangle });
    }
    meta
}

#[cfg(not(feature = "metadata"))]
pub(crate) fn clean_aperture_meta(_clean_aperture: Option<CropRectangle>) -> FrameMetaSet {
    FrameMetaSet::new()
}

impl TrackHeader {
    // the stream time the edit list presents, on the timeline `edited_sample` moves samples to
    fn edit_segment(&self) -> Segment {
        Segment {
            start: ticks_to_ns(self.media_start, self.timescale),
            stop: self.media_stop.map(|stop| {
                let stop_ns = ticks_to_ns(stop, self.timescale);
                stop_ns - self.cut_before_ns(stop_ns)
            }),
            ..Segment::new()
        }
    }

    // the last edit gap that starts at or before `media_ns`
    fn gap_from(&self, media_ns: u64) -> Option<&EditGap> {
        let later = self
            .edit_gaps
            .partition_point(|gap| gap.media_ns.start <= media_ns);
        later.checked_sub(1).map(|index| &self.edit_gaps[index])
    }

    // the media the edit gaps cut before `media_ns`, counting a gap it falls in up to it
    fn cut_before_ns(&self, media_ns: u64) -> u64 {
        self.gap_from(media_ns).map_or(0, |gap| {
            gap.earlier_gaps_ns + (media_ns.min(gap.media_ns.end) - gap.media_ns.start)
        })
    }

    // Moves a sample onto the timeline the edits present, with the gaps between
    // them cut out. A sample no edit presents still has to be decoded, as a
    // reference, so it goes past the segment's stop, where the decoder drops it.
    pub(crate) fn edited_sample(&self, sample: Sample) -> Sample {
        let Some(stop) = self.presentation_segment().stop else {
            return sample;
        };
        let gap = self
            .gap_from(sample.pts_ns)
            .filter(|gap| sample.pts_ns < gap.media_ns.end);
        let pts_ns = match gap {
            Some(gap) => stop.saturating_add(sample.pts_ns - gap.media_ns.start),
            None => sample.pts_ns - self.cut_before_ns(sample.pts_ns),
        };
        Sample {
            pts_ns,
            dts_ns: sample.dts_ns - self.cut_before_ns(sample.dts_ns),
            ..sample
        }
    }

    // An edit presents the samples its media span holds, decoded from the
    // keyframe before them. The ones decoded only as references go past the
    // segment's stop, where the decoder drops them.
    pub(crate) fn edited_samples(&self, samples: Vec<Sample>) -> Vec<Sample> {
        let Some(stop_ns) = self.presentation_segment().stop else {
            return samples;
        };
        if self.replayed_edits.is_empty() {
            return samples
                .into_iter()
                .map(|sample| self.edited_sample(sample))
                .collect();
        }
        let mut edited = Vec::new();
        for edit in &self.replayed_edits {
            let media_ns = ticks_to_ns(edit.media.start, self.timescale)
                ..ticks_to_ns(edit.media.end, self.timescale);
            let presented_ns = ticks_to_ns(
                self.media_start.saturating_add(edit.presented_from),
                self.timescale,
            );
            let presented = |sample: &Sample| media_ns.contains(&sample.pts_ns);
            let (Some(first), Some(last)) = (
                samples.iter().position(presented),
                samples.iter().rposition(presented),
            ) else {
                continue;
            };
            let from = samples[..=first]
                .iter()
                .rposition(|sample| sample.keyframe)
                .unwrap_or(0);
            for sample in &samples[from..=last] {
                let pts_ns = if presented(sample) {
                    presented_ns.saturating_add(sample.pts_ns - media_ns.start)
                } else {
                    stop_ns.saturating_add(sample.pts_ns)
                };
                edited.push(Sample {
                    annexb: sample.annexb.clone(),
                    pts_ns,
                    dts_ns: presented_ns
                        .saturating_add(sample.dts_ns)
                        .saturating_sub(media_ns.start),
                    ..*sample
                });
            }
        }
        edited
    }

    // An audio edit list trims the encoder's priming and padding off the
    // presentation: the segment starts past the one and stops before the other,
    // so the decoder clips them and a remux writes the start back as an edit
    // list. Opus is left out: its pre-skip rides in the `OpusHead`, which
    // `OpusDec` already trims.
    pub(crate) fn presentation_segment(&self) -> Segment {
        match self.kind {
            TrackKind::Audio { format, .. } if format != AudioFormat::Opus => self.edit_segment(),
            TrackKind::Video { .. } | TrackKind::RawVideo { .. } => self.edit_segment(),
            _ => Segment::new(),
        }
    }
}

// A track that replays media takes its samples out of file order, so the tracks then interleave by decode time.
pub(crate) fn edit_track_samples(
    tracks: &[TrackHeader],
    samples: Vec<(u32, Sample)>,
) -> Vec<(u32, Sample)> {
    let track_of = |track_id: u32| tracks.iter().find(|track| track.track_id == track_id);
    if tracks.iter().all(|track| track.replayed_edits.is_empty()) {
        return samples
            .into_iter()
            .map(|(track_id, sample)| match track_of(track_id) {
                Some(track) => (track_id, track.edited_sample(sample)),
                None => (track_id, sample),
            })
            .collect();
    }
    let mut by_track: Vec<(u32, Vec<Sample>)> = Vec::new();
    for (track_id, sample) in samples {
        match by_track.iter_mut().find(|(id, _)| *id == track_id) {
            Some((_, own)) => own.push(sample),
            None => by_track.push((track_id, Vec::from([sample]))),
        }
    }
    let mut edited: Vec<(u32, Sample)> = by_track
        .into_iter()
        .flat_map(|(track_id, own)| {
            let own = match track_of(track_id) {
                Some(track) => track.edited_samples(own),
                None => own,
            };
            own.into_iter().map(move |sample| (track_id, sample))
        })
        .collect();
    edited.sort_by_key(|(_, sample)| sample.dts_ns);
    edited
}

// PCM reaches no decoder, so the demuxers drop the samples outside the segment themselves
pub(crate) fn clip_stored_pcm(
    kind: &TrackKind,
    segment: &Segment,
    timing: FrameTiming,
    data: Vec<u8>,
) -> Result<Option<(FrameTiming, Vec<u8>)>, G2gError> {
    let TrackKind::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = kind
    else {
        return Ok(Some((timing, data)));
    };
    let Some(sample_bytes) = format.bytes_per_sample().filter(|_| is_pcm(*format)) else {
        return Ok(Some((timing, data)));
    };
    clip_pcm_to_segment(
        segment,
        timing,
        data,
        sample_bytes * usize::from(*channels),
        *sample_rate,
    )
}

/// Parse every forwardable (`vide` / `soun` / timed-text) track out of a `moov`
/// into a [`TrackHeader`]. The single-track [`parse_header`] reads only the first
/// `trak`; this walks them all (what an A/V `.mp4` carries). Tracks with an
/// unrecognized handler, or whose sample entry names a codec we do not read
/// (`tx3g` / `wvtt` / `stpp` text, `c608` / `c708` captions, and the video codecs
/// [`parse_video_entry`] handles are read, others decline) are skipped, not
/// errors; a malformed video / audio track fails the parse. Errors if no track is
/// forwardable.
pub(crate) fn parse_all_tracks(data: &[u8]) -> Result<Vec<TrackHeader>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    // mvhd lays out its version and timescale the way mdhd does
    let movie_timescale = find_box(moov, b"mvhd")
        .and_then(|mvhd| read_mdhd(mvhd).ok())
        .map(|header| header.timescale);
    let gapless = itunes_gapless(moov);
    let mut tracks = Vec::new();
    for (kind, trak) in boxes(moov) {
        if kind != b"trak" {
            continue;
        }
        if let Some(mut header) = parse_trak(trak)? {
            header.trex_defaults = trex_defaults(moov, header.track_id);
            let edits = edit_list(trak);
            let spans =
                movie_timescale.and_then(|movie| presented_spans(&edits, movie, header.timescale));
            let is_video = matches!(header.kind, TrackKind::Video { .. });
            let replayed = movie_timescale
                .filter(|_| spans.is_none() && is_video)
                .and_then(|movie| replayed_edits(&edits, movie, header.timescale));
            if let Some(spans) = spans {
                header.media_stop = spans.last().map(|span| span.end);
                header.edit_gaps = edit_gaps(&spans, header.timescale);
            } else if let Some((replayed, stop)) = replayed.and_then(|replayed| {
                let stop = replayed_media_stop(header.media_start, &replayed)?;
                Some((replayed, stop))
            }) {
                header.media_stop = Some(stop);
                header.replayed_edits = replayed;
            }
            // iTunes writes an AAC encoder's priming and padding as metadata instead of an edit list
            let is_audio = matches!(header.kind, TrackKind::Audio { .. });
            if let Some(presented) = gapless.clone().filter(|_| is_audio && edits.is_empty()) {
                header.media_start = presented.start;
                header.media_stop = Some(presented.end);
            }
            tracks.push(header);
        }
    }
    if tracks.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(tracks)
}

/// Parse one `trak`. Returns `None` for a handler we do not forward (skip it),
/// `Err` for a malformed video / audio track.
fn parse_trak(trak: &[u8]) -> Result<Option<TrackHeader>, G2gError> {
    let TrackHeaderBox {
        track_id,
        width,
        height,
    } = read_tkhd(find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?)?;
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let timescale = read_mdhd(find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?)?.timescale;

    // hdlr handler_type at payload offset 8 (4 version/flags + 4 pre_defined)
    // selects how to read the sample entry.
    let hdlr = find_box(mdia, b"hdlr").ok_or(G2gError::CapsMismatch)?;
    let handler = hdlr.get(8..12).ok_or(G2gError::CapsMismatch)?;

    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    let stsd = find_box(stbl, b"stsd").ok_or(G2gError::CapsMismatch)?;
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;

    let clean_aperture = match handler {
        b"vide" | b"pict" => clean_aperture(entries),
        _ => None,
    };
    let (kind, cenc) = match handler {
        // A video sample entry we have no reader for is skipped like an unknown
        // handler: one unsupported track must not fail the whole file.
        // `pict` is a HEIF / AVIF image sequence, coded like a video track
        b"vide" | b"pict" => match parse_video_entry(entries, width, height)? {
            Some(entry) => entry,
            None => return Ok(None),
        },
        b"soun" => match parse_audio_entry(entries, timescale)? {
            Some(entry) => entry,
            None => return Ok(None),
        },
        // A subtitle / timed-text handler (`text` = 3GPP/QuickTime timed text,
        // `sbtl` / `subt` = MP4 / ISO subtitle). Forwarded only when its sample
        // entry is one we de-frame (`tx3g`); an unrecognized text codec is skipped.
        b"text" | b"sbtl" | b"subt" => match parse_text_entry(entries) {
            Some(kind) => (kind, None),
            None => return Ok(None),
        },
        // A closed-caption handler (`clcp`), whose sample entry names the caption
        // carriage: `c608` / `c708` are read, anything else is skipped.
        b"clcp" => match parse_cc_entry(entries) {
            Some(kind) => (kind, None),
            None => return Ok(None),
        },
        _ => return Ok(None), // hint / metadata / unknown handler: not forwarded
    };
    Ok(Some(TrackHeader {
        track_id,
        timescale,
        kind,
        cenc: cenc_track(cenc, stbl)?,
        tags: parse_ilst_tags(trak),
        media_start: edit_media_start(&edit_list(trak)),
        media_stop: None,
        edit_gaps: Vec::new(),
        replayed_edits: Vec::new(),
        trex_defaults: SampleDefaults::default(),
        clean_aperture,
    }))
}

// trex payload: version/flags, track_ID, sample description index, then the defaults
const TREX_TRACK_ID_AT: usize = 4;
const TREX_DURATION_AT: usize = 12;
const TREX_SIZE_AT: usize = 16;
const TREX_FLAGS_AT: usize = 20;

fn trex_defaults(moov: &[u8], track_id: u32) -> SampleDefaults {
    let Some(mvex) = find_box(moov, b"mvex") else {
        return SampleDefaults::default();
    };
    boxes(mvex)
        .filter(|(kind, _)| *kind == b"trex")
        .find(|(_, trex)| be32(trex, TREX_TRACK_ID_AT).ok() == Some(track_id))
        .and_then(|(_, trex)| {
            Some(SampleDefaults {
                duration: be32(trex, TREX_DURATION_AT).ok()?,
                size: be32(trex, TREX_SIZE_AT).ok()?,
                flags: Some(be32(trex, TREX_FLAGS_AT).ok()?),
            })
        })
        .unwrap_or_default()
}

/// The `media_time` of a `trak`'s first media edit (`edts/elst`), in media
/// timescale ticks: where its presentation starts, which is how MP4 records an
/// encoder's priming. `0` for no edit list, no media edit (an empty edit,
/// `media_time` -1, is a presentation delay, not a trim), or a malformed box: the
/// edit list is untrusted, so anything unreadable falls back to presenting from
/// the start.
fn edit_media_start(edits: &[Edit]) -> u64 {
    edits
        .iter()
        .find(|edit| edit.media_time >= 0)
        .map_or(0, |edit| edit.media_time as u64)
}

// The media spans the edits after the leading empty ones present, in media
// ticks. `None` unless each is a media edit with a duration that starts at or
// after the end of the one before: a zero segment_duration is a fragmented
// file's "until the end".
fn presented_spans(
    edits: &[Edit],
    movie_timescale: u32,
    media_timescale: u32,
) -> Option<Vec<Range<u64>>> {
    let mut spans: Vec<Range<u64>> = Vec::new();
    for edit in edits.iter().skip_while(|edit| edit.media_time < 0) {
        let media_time = u64::try_from(edit.media_time).ok()?;
        if edit.segment_duration == 0 {
            return None;
        }
        let duration = segment_media_ticks(edit, movie_timescale, media_timescale)?;
        let end = media_time.checked_add(duration)?;
        if spans.last().is_some_and(|last| media_time < last.end) {
            return None;
        }
        spans.push(media_time..end);
    }
    (!spans.is_empty()).then_some(spans)
}

// segment_duration counts movie ticks, rounded to the nearest media tick
fn segment_media_ticks(edit: &Edit, movie_timescale: u32, media_timescale: u32) -> Option<u64> {
    let duration = (u128::from(edit.segment_duration) * u128::from(media_timescale)
        + u128::from(movie_timescale / 2))
    .checked_div(u128::from(movie_timescale))?;
    u64::try_from(duration).ok()
}

// The media edits after the leading empty ones, laid out one after another,
// for a list presented_spans cannot read because an edit goes back in the media.
fn replayed_edits(
    edits: &[Edit],
    movie_timescale: u32,
    media_timescale: u32,
) -> Option<Vec<ReplayedEdit>> {
    let mut replayed: Vec<ReplayedEdit> = Vec::new();
    let mut presented_from = 0u64;
    for edit in edits.iter().skip_while(|edit| edit.media_time < 0) {
        let media_time = u64::try_from(edit.media_time).ok()?;
        let duration = segment_media_ticks(edit, movie_timescale, media_timescale)
            .filter(|duration| *duration > 0)?;
        replayed.push(ReplayedEdit {
            presented_from,
            media: media_time..media_time.checked_add(duration)?,
        });
        presented_from = presented_from.checked_add(duration)?;
    }
    (!replayed.is_empty()).then_some(replayed)
}

fn replayed_media_stop(media_start: u64, replayed: &[ReplayedEdit]) -> Option<u64> {
    let last = replayed.last()?;
    media_start
        .checked_add(last.presented_from)?
        .checked_add(last.media.end - last.media.start)
}

fn edit_gaps(spans: &[Range<u64>], timescale: u32) -> Vec<EditGap> {
    let mut gaps = Vec::new();
    let mut earlier_gaps_ns = 0;
    for pair in spans.windows(2) {
        let media_ns = ticks_to_ns(pair[0].end, timescale)..ticks_to_ns(pair[1].start, timescale);
        if media_ns.is_empty() {
            continue;
        }
        let gap_ns = media_ns.end - media_ns.start;
        gaps.push(EditGap {
            media_ns,
            earlier_gaps_ns,
        });
        earlier_gaps_ns += gap_ns;
    }
    gaps
}

#[derive(Debug)]
struct Edit {
    // movie timescale ticks
    segment_duration: u64,
    // media timescale ticks, -1 for an empty edit
    media_time: i64,
}

// the entries that fit the box, as ffmpeg reads them, none when it is missing or malformed
fn edit_list(trak: &[u8]) -> Vec<Edit> {
    let Some(elst) = find_path(trak, &[b"edts", b"elst"]) else {
        return Vec::new();
    };
    let Ok(entry_count) = be32(elst, 4) else {
        return Vec::new();
    };
    // v0: u32 segment_duration + i32 media_time; v1: u64 + i64, each then a
    // 32-bit rate. The entries start after version/flags (4) and entry_count (4).
    let version_1 = match elst.first() {
        Some(0) => false,
        Some(1) => true,
        _ => return Vec::new(),
    };
    let entry_len = if version_1 { 20 } else { 12 };
    let read_entry = |at: usize| {
        Some(if version_1 {
            Edit {
                segment_duration: be64(elst, at).ok()?,
                media_time: be64(elst, at + 8).ok()? as i64,
            }
        } else {
            Edit {
                segment_duration: u64::from(be32(elst, at).ok()?),
                media_time: i64::from(be32(elst, at + 4).ok()? as i32),
            }
        })
    };
    (0..entry_count as usize)
        .map_while(|index| read_entry(8 + index * entry_len))
        .collect()
}

// visual sample entry: the fixed fields before its child boxes
const VISUAL_SAMPLE_ENTRY_CHILDREN_AT: usize = 78;

// the first sample entry's `clap` against the entry's own size, None for no box or one that cuts nothing
fn clean_aperture(entries: &[u8]) -> Option<CropRectangle> {
    let (_, entry) = boxes(entries).next()?;
    let clap = find_box(entry.get(VISUAL_SAMPLE_ENTRY_CHILDREN_AT..)?, b"clap")?;
    let dimension = |at: usize| {
        let bytes = entry.get(at..at + 2)?;
        Some(u32::from(u16::from_be_bytes([bytes[0], bytes[1]])))
    };
    let width = dimension(VISUAL_SAMPLE_ENTRY_WIDTH_AT)?;
    let height = dimension(VISUAL_SAMPLE_ENTRY_HEIGHT_AT)?;
    match clap_rectangle(clap, width, height) {
        Ok(rectangle) => rectangle,
        Err(_) => {
            g2g_warn!(
                Target::category("qtdemux"),
                "ignored a clap box that does not fit the {width}x{height} picture"
            );
            None
        }
    }
}

// clap holds numerator and denominator pairs: aperture width, aperture height, horizontal and vertical centre offset
fn clap_rectangle(clap: &[u8], width: u32, height: u32) -> Result<Option<CropRectangle>, G2gError> {
    let field = |index: usize| be32(clap, index * 4).map(|value| value as i32);
    let fraction =
        |index: usize| -> Result<(i32, i32), G2gError> { Ok((field(index)?, field(index + 1)?)) };
    let (left, right) =
        clap_axis(width, fraction(0)?, fraction(4)?).ok_or(G2gError::CapsMismatch)?;
    let (top, bottom) =
        clap_axis(height, fraction(2)?, fraction(6)?).ok_or(G2gError::CapsMismatch)?;
    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        return Ok(None);
    }
    Ok(Some(CropRectangle {
        left,
        top,
        width: width - left - right,
        height: height - top - bottom,
    }))
}

// the pixels one axis of the aperture cuts before and after it, each edge truncated as ffmpeg's mov_read_clap does
fn clap_axis(picture: u32, aperture: (i32, i32), offset: (i32, i32)) -> Option<(u32, u32)> {
    let (aperture_numerator, aperture_denominator) =
        (i128::from(aperture.0), i128::from(aperture.1));
    let (offset_numerator, offset_denominator) = (i128::from(offset.0), i128::from(offset.1));
    if aperture_numerator < 0 || aperture_denominator <= 0 || offset_denominator <= 0 {
        return None;
    }
    let picture = i128::from(picture);
    let denominator = 2 * aperture_denominator * offset_denominator;
    let shift = 2 * aperture_denominator * offset_numerator;
    let first = ((picture * aperture_denominator - aperture_numerator) * offset_denominator
        + shift)
        / denominator;
    let last = (((picture - 2) * aperture_denominator + aperture_numerator) * offset_denominator
        + shift)
        / denominator;
    if first < 0 || last < 0 || last > picture - 1 {
        return None;
    }
    let after = picture - 1 - last;
    if first + after >= picture {
        return None;
    }
    Some((u32::try_from(first).ok()?, u32::try_from(after).ok()?))
}

/// The out-of-band config from an `av01` sample entry's `av1C` record (M779):
/// the trailing configOBUs (typically the sequence header), verbatim, as the
/// one "parameter set". Samples are plain low-overhead OBUs, so an empty
/// configOBUs just means the sequence header rides in-band.
fn parse_av1c_config(av01: &[u8]) -> Result<Vec<Vec<u8>>, G2gError> {
    let children = av01.get(78..).ok_or(G2gError::CapsMismatch)?;
    let av1c = find_box(children, b"av1C").ok_or(G2gError::CapsMismatch)?;
    // marker/version, profile/level, tier/depth flags, delay byte: 4 fixed
    // bytes, then configOBUs.
    let config = av1c.get(4..).unwrap_or_default();
    Ok(if config.is_empty() {
        Vec::new()
    } else {
        Vec::from([config.to_vec()])
    })
}

/// Read a video sample entry (`avc1` / `avc3` / `hvc1` / `hev1` / `mp4v` / `av01` /
/// `vp08` / `jpeg`, or the encrypted `encv`)
/// into a [`TrackKind::Video`] plus the cbcs `cenc` defaults for an encrypted
/// track. An `encv` carries the original codec config (`avcC` / `hvcC`) alongside
/// a `sinf` (original format + `cbcs` scheme + `tenc`), the same shape
/// [`parse_header`] reads. Returns `None` for an entry that is well-formed but
/// names a codec with no reader here (VP6 in MOV, say), so that track is
/// skipped; malformed entry data is still an error.
fn parse_video_entry(
    entries: &[u8],
    width: u32,
    height: u32,
) -> Result<Option<(TrackKind, Option<CencDefaults>)>, G2gError> {
    // ahead of a clear entry, so ciphertext never decodes as clear
    let (codec, param_sets, cenc) = if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_sinf(sinf)?;
        let frma = find_box(sinf, b"frma").ok_or(G2gError::CapsMismatch)?;
        let (codec, param_sets) = match frma.get(0..4) {
            Some(b"avc1") => {
                let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H264, parse_avcc(avcc)?.parameter_sets)
            }
            Some(b"hvc1") | Some(b"hev1") => {
                let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H265, parse_hvcc(hvcc)?.parameter_sets)
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        (codec, param_sets, Some(cenc))
    } else if let Some(avc1) = find_box(entries, b"avc1").or_else(|| find_box(entries, b"avc3")) {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H264, parse_avcc(avcc)?.parameter_sets, None)
    } else if let Some(hvc1) = find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1")) {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?.parameter_sets, None)
    } else if find_box(entries, b"vp08").is_some() {
        (VideoCodec::Vp8, Vec::new(), None)
    } else if find_box(entries, b"jpeg")
        .or_else(|| find_box(entries, b"AVDJ"))
        .is_some()
    {
        (VideoCodec::Mjpeg, Vec::new(), None)
    } else if let Some(mp4v) = find_box(entries, b"mp4v") {
        match mp4v_codec(mp4v)? {
            Some((codec, sets)) => (codec, sets, None),
            // a valid entry we have no reader for
            None => return Ok(None),
        }
    } else if let Some(av01) = find_box(entries, b"av01") {
        (VideoCodec::Av1, parse_av1c_config(av01)?, None)
    } else if let Some(codec) = boxes(entries).find_map(|(fourcc, _)| mezzanine_codec(*fourcc)) {
        (codec, Vec::new(), None)
    } else if let Some((format, entry)) = boxes(entries).find_map(|(fourcc, entry)| {
        Some((crate::uncompressedvideo::raw_video_format(*fourcc)?, entry))
    }) {
        return Ok(Some((
            raw_video_track(format, RawVideoStorage::Tight, entry)?,
            None,
        )));
    } else if let Some((format, storage, entry)) = find_box(entries, b"raw ").and_then(|entry| {
        let depth = entry.get(VISUAL_SAMPLE_ENTRY_DEPTH_AT..VISUAL_SAMPLE_ENTRY_DEPTH_AT + 2)?;
        let (format, storage) = quicktime_raw_video(u16::from_be_bytes([depth[0], depth[1]]))?;
        Some((format, storage, entry))
    }) {
        return Ok(Some((raw_video_track(format, storage, entry)?, None)));
    } else if visual_sample_entry(entries) {
        // A well-formed VisualSampleEntry naming a codec we do not read: decline
        // so the rest of the file still demuxes.
        return Ok(unread_video_entry(entries, width, height));
    } else {
        return Err(G2gError::CapsMismatch);
    };
    Ok(Some((
        TrackKind::Video {
            codec,
            width,
            height,
            param_sets,
        },
        cenc,
    )))
}

// ISO/IEC 14496-1 objectTypeIndication values of an `mp4v` entry's decoder config
const MPEG4_VISUAL_OBJECT_TYPE: u8 = 0x20;
// the simple, main, SNR, spatial, high and 4:2:2 profiles
const MPEG2_VISUAL_OBJECT_TYPES: core::ops::RangeInclusive<u8> = 0x60..=0x65;
const MPEG1_VISUAL_OBJECT_TYPE: u8 = 0x6A;

// a sample entry's codec and the config blobs it carries out of band
type EntryCodec = (VideoCodec, Vec<Vec<u8>>);

// None for an objectTypeIndication g2g has no codec for (ffmpeg writes MJPEG-in-MP4 as 0x6C)
fn mp4v_codec(mp4v: &[u8]) -> Result<Option<EntryCodec>, G2gError> {
    let children = mp4v
        .get(VISUAL_SAMPLE_ENTRY_CHILDREN_AT..)
        .ok_or(G2gError::CapsMismatch)?;
    let esds = find_box(children, b"esds").ok_or(G2gError::CapsMismatch)?;
    let (object_type, decoder_config) = parse_esds_decoder_config(esds)?;
    let codec = match object_type {
        MPEG4_VISUAL_OBJECT_TYPE => VideoCodec::Mpeg4Part2,
        // one codec in g2g, as one decoder plays both
        MPEG1_VISUAL_OBJECT_TYPE => VideoCodec::Mpeg2,
        object_type if MPEG2_VISUAL_OBJECT_TYPES.contains(&object_type) => VideoCodec::Mpeg2,
        _ => return Ok(None),
    };
    // the VOL or sequence header, empty when the stream carries it in-band
    let sets = match decoder_config.is_empty() {
        true => Vec::new(),
        false => Vec::from([decoder_config]),
    };
    Ok(Some((codec, sets)))
}

// the intra-only editing codecs, each frame decodable alone
fn mezzanine_codec(fourcc: [u8; 4]) -> Option<VideoCodec> {
    if let Some(variant) = ProResVariant::from_fourcc(fourcc) {
        return Some(VideoCodec::ProRes { variant });
    }
    matches!(&fourcc, b"AVdn" | b"AVdh").then_some(VideoCodec::DnxHd)
}

// VisualSampleEntry: reserved, data_reference_index and pre_defined fields, then width and height
const VISUAL_SAMPLE_ENTRY_WIDTH_AT: usize = 24;
const VISUAL_SAMPLE_ENTRY_HEIGHT_AT: usize = 26;
// past the resolutions, frame count and 32-byte compressor name
const VISUAL_SAMPLE_ENTRY_DEPTH_AT: usize = 74;

fn raw_video_track(
    format: RawVideoFormat,
    storage: RawVideoStorage,
    entry: &[u8],
) -> Result<TrackKind, G2gError> {
    let dimension = |at: usize| -> Result<u32, G2gError> {
        let bytes = entry.get(at..at + 2).ok_or(G2gError::CapsMismatch)?;
        match u16::from_be_bytes([bytes[0], bytes[1]]) {
            0 => Err(G2gError::CapsMismatch),
            pixels => Ok(u32::from(pixels)),
        }
    };
    Ok(TrackKind::RawVideo {
        format,
        width: dimension(VISUAL_SAMPLE_ENTRY_WIDTH_AT)?,
        height: dimension(VISUAL_SAMPLE_ENTRY_HEIGHT_AT)?,
        storage,
    })
}

// a sample as the graph's tight top-down frame, one too short for that left as stored for the demuxer to drop
fn raw_video_sample(stored: &[u8], framing: RawVideoFraming) -> Vec<u8> {
    let RawVideoFraming {
        format,
        storage,
        width,
        height,
    } = framing;
    copy_raw_video_frame(format, storage, width, height, stored).unwrap_or_else(|| stored.to_vec())
}

#[derive(Debug, Clone, Copy)]
struct RawVideoFraming {
    format: RawVideoFormat,
    storage: RawVideoStorage,
    width: u32,
    height: u32,
}

// a sample entry no reader here takes goes on under its fourcc, its whole entry leading the samples
#[cfg(feature = "ffmpeg-long-tail")]
fn unread_video_entry(
    entries: &[u8],
    width: u32,
    height: u32,
) -> Option<(TrackKind, Option<CencDefaults>)> {
    let (fourcc, payload) = boxes(entries).next()?;
    let kind = TrackKind::Video {
        codec: VideoCodec::Tagged(mov_tag(fourcc)),
        width,
        height,
        param_sets: Vec::from([crate::mp4box::mp4_box(fourcc, payload)]),
    };
    Some((kind, None))
}

#[cfg(not(feature = "ffmpeg-long-tail"))]
fn unread_video_entry(
    _entries: &[u8],
    _width: u32,
    _height: u32,
) -> Option<(TrackKind, Option<CencDefaults>)> {
    None
}

#[cfg(feature = "ffmpeg-long-tail")]
fn mov_tag(fourcc: &[u8; 4]) -> g2g_core::CodecTag {
    g2g_core::CodecTag {
        namespace: g2g_core::TagNamespace::Mov,
        tag: u32::from_le_bytes(*fourcc),
    }
}

// a sound description that does not parse leaves the track skipped, as before tagged codecs
#[cfg(feature = "ffmpeg-long-tail")]
fn unread_audio_entry(
    entries: &[u8],
    timescale: u32,
) -> Result<Option<(TrackKind, Option<CencDefaults>)>, G2gError> {
    let Some((fourcc, payload)) = boxes(entries).next() else {
        return Ok(None);
    };
    let tag = mov_tag(fourcc);
    let Ok(description) = SoundDescription::parse(payload, tag.tag) else {
        return Ok(None);
    };
    let Some(channels) = u8::try_from(description.channels)
        .ok()
        .filter(|channels| *channels > 0)
    else {
        return Ok(None);
    };
    let kind = TrackKind::Audio {
        format: AudioFormat::Tagged(tag),
        channels,
        sample_rate: match description.sample_rate {
            0 => timescale,
            rate => rate,
        },
        config: crate::mp4box::mp4_box(fourcc, payload),
        stored_pcm: None,
    };
    Ok(Some((kind, None)))
}

#[cfg(not(feature = "ffmpeg-long-tail"))]
fn unread_audio_entry(
    _entries: &[u8],
    _timescale: u32,
) -> Result<Option<(TrackKind, Option<CencDefaults>)>, G2gError> {
    Ok(None)
}

// The QuickTime sound description fields mov.c reads to cut a track into
// packets, lay out its PCM and set up its decoder. Offsets count from the entry payload.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SoundDescription {
    pub version: u16,
    pub channels: u16,
    pub sample_bits: u16,
    // 0 when the entry leaves the rate to the media timescale
    pub sample_rate: u32,
    pub samples_per_frame: u32,
    pub bytes_per_frame: u32,
    // the version 2 formatSpecificFlags, 0 for the older versions
    pub format_flags: u32,
    pub children_at: usize,
}

impl SoundDescription {
    pub(crate) fn parse(payload: &[u8], fourcc: u32) -> Result<Self, G2gError> {
        const V1_SAMPLES_PER_FRAME: usize = 28;
        const V1_BYTES_PER_FRAME: usize = 36;
        const V2_SAMPLE_RATE: usize = 32;
        const V2_CHANNELS: usize = 40;
        const V2_SAMPLE_BITS: usize = 48;
        const V2_FORMAT_FLAGS: usize = 52;
        const V2_BYTES_PER_FRAME: usize = 56;
        const V2_SAMPLES_PER_FRAME: usize = 60;
        let mut description = SoundDescription {
            version: sound_entry_u16(payload, SOUND_ENTRY_VERSION)?,
            channels: sound_entry_u16(payload, SOUND_ENTRY_CHANNELS)?,
            sample_bits: sound_entry_u16(payload, SOUND_ENTRY_SAMPLE_BITS)?,
            sample_rate: u32::from(sound_entry_u16(payload, SOUND_ENTRY_SAMPLE_RATE)?),
            samples_per_frame: 0,
            bytes_per_frame: 0,
            format_flags: 0,
            children_at: sound_entry_children_at(payload)?,
        };
        match description.version {
            1 => {
                description.samples_per_frame = be32(payload, V1_SAMPLES_PER_FRAME)?;
                description.bytes_per_frame = be32(payload, V1_BYTES_PER_FRAME)?;
            }
            SOUND_ENTRY_VERSION_2 => {
                let rate = f64::from_bits(be64(payload, V2_SAMPLE_RATE)?);
                // a NaN or out-of-range rate saturates, and a zero one falls back to the timescale
                description.sample_rate = rate as u32;
                description.channels = u16::try_from(be32(payload, V2_CHANNELS)?)
                    .map_err(|_| G2gError::CapsMismatch)?;
                description.sample_bits = u16::try_from(be32(payload, V2_SAMPLE_BITS)?)
                    .map_err(|_| G2gError::CapsMismatch)?;
                description.format_flags = be32(payload, V2_FORMAT_FLAGS)?;
                description.bytes_per_frame = be32(payload, V2_BYTES_PER_FRAME)?;
                description.samples_per_frame = be32(payload, V2_SAMPLES_PER_FRAME)?;
            }
            _ => {}
        }
        // mov.c's fixed framing for the codecs that predate version 1, which wins over the entry's
        let channels = u32::from(description.channels);
        let fixed = match &fourcc.to_le_bytes() {
            b"MAC3" => Some((6, channels.saturating_mul(2))),
            b"MAC6" => Some((6, channels)),
            b"ima4" => Some((64, channels.saturating_mul(34))),
            b"agsm" => Some((160, 33)),
            _ => None,
        };
        if let Some((samples, bytes)) = fixed {
            description.samples_per_frame = samples;
            description.bytes_per_frame = bytes;
        }
        Ok(description)
    }
}

/// Whether `entries` (an `stsd`'s sample-entry list) holds at least one entry
/// well-formed enough to be a VisualSampleEntry: a parseable box carrying its 78
/// fixed bytes. Truncated or garbage entry data is not, and fails the parse
/// rather than being silently skipped.
fn visual_sample_entry(entries: &[u8]) -> bool {
    boxes(entries).any(|(_, payload)| payload.len() >= 78)
}

/// Read an audio sample entry into a [`TrackKind::Audio`] plus the cbcs `cenc`
/// defaults for an encrypted track: AAC (`mp4a`/`esds`, or the encrypted
/// `enca`) or Opus (`Opus`/`dOps`, M767). The sample rate is the media
/// timescale (matching `Mp4AudioSrc`). `None` for a well-formed entry naming a
/// codec with no reader here, so that track is skipped.
fn parse_audio_entry(
    entries: &[u8],
    timescale: u32,
) -> Result<Option<(TrackKind, Option<CencDefaults>)>, G2gError> {
    // Opus: the `dOps` carries the same fields as an `OpusHead`, so it is
    // rebuilt into one and forwarded in-band (M791), the convention `OggDemux`
    // already uses, so the decoder trims the pre-skip and a remux keeps it. The
    // `dOps` OutputChannelCount is authoritative over the sample-entry
    // channelcount. A `dOps` that does not parse leaves the track without config
    // (playable, untrimmed) rather than failing the whole file.
    if let Some(opus) = find_box(entries, b"Opus") {
        let entry_channels = u16::from_be_bytes(
            opus.get(16..18)
                .ok_or(G2gError::CapsMismatch)?
                .try_into()
                .expect("2 bytes"),
        ) as u8;
        let config = find_box(opus.get(28..).unwrap_or(&[]), b"dOps")
            .and_then(opus_head_from_dops)
            .unwrap_or_default();
        let channels = parse_opus_head(&config)
            .map(|(c, _)| c)
            .unwrap_or(entry_channels);
        if channels == 0 {
            return Err(G2gError::CapsMismatch);
        }
        return Ok(Some((
            TrackKind::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate: timescale,
                config,
                stored_pcm: None,
            },
            None,
        )));
    }
    if let Some((pcm, description)) = quicktime_pcm_entry(entries) {
        // a layout no AudioFormat carries skips the track, which `unsupported_pcm_layout` names
        let Ok((format, stored_pcm)) = pcm else {
            return Ok(None);
        };
        let kind = quicktime_pcm_track(format, stored_pcm, &description)?;
        return Ok(Some((kind, None)));
    }
    if let Some(vorbis) = find_box(entries, b"msVo") {
        return Ok(Some((vorbis_track(vorbis, timescale)?, None)));
    }
    if let Some(mp3) = find_box(entries, b".mp3") {
        let channels = sound_entry_channels(mp3)?;
        return Ok(Some((
            TrackKind::Audio {
                format: AudioFormat::Mp3,
                channels,
                sample_rate: timescale,
                config: Vec::new(),
                stored_pcm: None,
            },
            None,
        )));
    }
    let (entry, cenc) = match (find_box(entries, b"mp4a"), find_box(entries, b"enca")) {
        (Some(mp4a), _) => (mp4a, None),
        (None, Some(enca)) => {
            let children = enca.get(28..).ok_or(G2gError::CapsMismatch)?;
            let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
            (enca, Some(parse_sinf(sinf)?))
        }
        // a codec we have no reader for (ALAC, ...): skip the track
        (None, None) if boxes(entries).next().is_some() => {
            return unread_audio_entry(entries, timescale)
        }
        (None, None) => return Err(G2gError::CapsMismatch),
    };
    let channels = sound_entry_channels(entry)?;
    let children = entry
        .get(sound_entry_children_at(entry)?..)
        .ok_or(G2gError::CapsMismatch)?;
    // QuickTime nests the esds in a `wave` box
    let esds = find_box(children, b"esds")
        .or_else(|| find_path(children, &[b"wave", b"esds"]))
        .ok_or(G2gError::CapsMismatch)?;
    if MPEG_AUDIO_OBJECT_TYPES.contains(&parse_esds_decoder_config(esds)?.0) {
        return Ok(Some((
            TrackKind::Audio {
                format: AudioFormat::Mp3,
                channels,
                sample_rate: timescale,
                config: Vec::new(),
                stored_pcm: None,
            },
            cenc,
        )));
    }
    let config = parse_esds(esds)?;
    Ok(Some((
        TrackKind::Audio {
            format: AudioFormat::Aac,
            channels,
            sample_rate: timescale,
            config,
            stored_pcm: None,
        },
        cenc,
    )))
}

// SoundDescription fields, from the entry payload start
const SOUND_ENTRY_VERSION: core::ops::Range<usize> = 8..10;
const SOUND_ENTRY_CHANNELS: core::ops::Range<usize> = 16..18;
const SOUND_ENTRY_SAMPLE_BITS: core::ops::Range<usize> = 18..20;
// the integer half of the 16.16 sample rate
const SOUND_ENTRY_SAMPLE_RATE: core::ops::Range<usize> = 24..26;
// version 2 moves the real channel count, depth and rate into fields of its own
const SOUND_ENTRY_VERSION_2: u16 = 2;

fn sound_entry_u16(entry: &[u8], field: core::ops::Range<usize>) -> Result<u16, G2gError> {
    let bytes = entry.get(field).ok_or(G2gError::CapsMismatch)?;
    Ok(u16::from_be_bytes(bytes.try_into().expect("2 bytes")))
}

fn sound_entry_channels(entry: &[u8]) -> Result<u8, G2gError> {
    match u8::try_from(sound_entry_u16(entry, SOUND_ENTRY_CHANNELS)?) {
        Ok(0) | Err(_) => Err(G2gError::CapsMismatch),
        Ok(channels) => Ok(channels),
    }
}

// the graph format a QuickTime PCM entry carries, or its layout when no AudioFormat does
type PcmGraphFormat = Result<(AudioFormat, StoredPcm), PcmLayout>;

// the first PCM sample entry in an `stsd`'s list
fn quicktime_pcm_entry(entries: &[u8]) -> Option<(PcmGraphFormat, SoundDescription)> {
    boxes(entries).find_map(|(fourcc, entry)| {
        let description = SoundDescription::parse(entry, u32::from_le_bytes(*fourcc)).ok()?;
        Some((
            quicktime_pcm_format(fourcc, &description, entry)?,
            description,
        ))
    })
}

// the PCM layout of a QuickTime sound entry that no AudioFormat carries
pub(crate) fn unsupported_pcm_layout(entries: &[u8]) -> Option<PcmLayout> {
    quicktime_pcm_entry(entries)?.0.err()
}

// a QuickTime PCM track, at its sound description's rate rather than the media timescale
fn quicktime_pcm_track(
    format: AudioFormat,
    stored_pcm: StoredPcm,
    description: &SoundDescription,
) -> Result<TrackKind, G2gError> {
    let channels = u8::try_from(description.channels)
        .ok()
        .filter(|channels| *channels > 0)
        .ok_or(G2gError::CapsMismatch)?;
    if description.sample_rate == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(TrackKind::Audio {
        format,
        channels,
        sample_rate: description.sample_rate,
        config: Vec::new(),
        stored_pcm: Some(stored_pcm),
    })
}

// QuickTime names its PCM entries with AIFC's compression types, `twos`, `raw `, the `in` / `fl` types and `lpcm`
fn quicktime_pcm_format(
    fourcc: &[u8; 4],
    description: &SoundDescription,
    entry: &[u8],
) -> Option<PcmGraphFormat> {
    const RAW_UNSIGNED_BITS: u16 = 8;
    const RAW_BIG_ENDIAN_BITS: u16 = 16;
    let bits = description.sample_bits;
    let aiff = |compression: [u8; 4]| {
        let (format, big_endian) = audio_from_comm(bits, Some(compression))?;
        Some(Ok((format, StoredPcm::Aiff { big_endian })))
    };
    // the fourcc fixes the sample type, which ffmpeg writes in a version 1 entry whose bit depth field says 16
    let typed = |bits: u32, float: bool| {
        let layout = PcmLayout {
            bits,
            float,
            signed: true,
            big_endian: !enda_little_endian(entry, description),
            packed: true,
        };
        Some(layout.graph_format())
    };
    match fourcc {
        b"raw " if bits == RAW_UNSIGNED_BITS => {
            Some(Ok((AudioFormat::PcmU8, StoredPcm::Unsigned8)))
        }
        // AIFC's `NONE` is signed big-endian
        b"raw " if bits == RAW_BIG_ENDIAN_BITS => aiff(*b"NONE"),
        b"twos" => aiff(*b"NONE"),
        b"sowt" | b"alaw" | b"ulaw" => aiff(*fourcc),
        b"in24" => typed(24, false),
        b"in32" => typed(32, false),
        b"fl32" => typed(32, true),
        b"fl64" => typed(64, true),
        b"lpcm" => Some(lpcm_layout(description).graph_format()),
        _ => None,
    }
}

// an `enda` atom inside the entry's `wave` atom holding 1 marks little-endian samples, as mov.c reads it
fn enda_little_endian(entry: &[u8], description: &SoundDescription) -> bool {
    entry
        .get(description.children_at..)
        .and_then(|children| find_path(children, &[b"wave", b"enda"]))
        .is_some_and(|enda| enda.get(1) == Some(&1))
}

// kAudioFormatFlagIsFloat, IsBigEndian and IsSignedInteger in a version 2 entry's formatSpecificFlags
const LPCM_FLAG_FLOAT: u32 = 0x1;
const LPCM_FLAG_BIG_ENDIAN: u32 = 0x2;
const LPCM_FLAG_SIGNED: u32 = 0x4;

fn lpcm_layout(description: &SoundDescription) -> PcmLayout {
    let bits = u32::from(description.sample_bits);
    let flags = description.format_flags;
    let frame_bytes = bits / 8 * u32::from(description.channels);
    PcmLayout {
        bits,
        float: flags & LPCM_FLAG_FLOAT != 0,
        signed: flags & LPCM_FLAG_SIGNED != 0,
        big_endian: flags & LPCM_FLAG_BIG_ENDIAN != 0,
        // the packet sizes say whether samples sit back to back, whatever the IsPacked flag claims
        packed: bits % 8 == 0
            && description.bytes_per_frame == frame_bytes
            && description.samples_per_frame == 1,
    }
}

// how a QuickTime PCM entry stores its samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PcmLayout {
    bits: u32,
    float: bool,
    signed: bool,
    big_endian: bool,
    // one frame per packet, each frame its samples back to back
    packed: bool,
}

impl PcmLayout {
    fn graph_format(self) -> PcmGraphFormat {
        let format = match (self.float, self.signed, self.bits) {
            _ if !self.packed => return Err(self),
            (true, _, 32) => AudioFormat::PcmF32Le,
            (false, _, 8) => AudioFormat::PcmU8,
            (false, true, 16) => AudioFormat::PcmS16Le,
            (false, true, 24) => AudioFormat::PcmS24Le,
            (false, true, 32) => AudioFormat::PcmS32Le,
            _ => return Err(self),
        };
        let stored = match (format, self.signed) {
            (AudioFormat::PcmU8, false) => StoredPcm::Unsigned8,
            _ => StoredPcm::Aiff {
                big_endian: self.big_endian,
            },
        };
        Ok((format, stored))
    }
}

impl core::fmt::Display for PcmLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match (self.float, self.signed) {
            (true, _) => "float",
            (false, true) => "signed",
            (false, false) => "unsigned",
        };
        let order = if self.big_endian {
            "big-endian"
        } else {
            "little-endian"
        };
        write!(f, "{}-bit {kind} {order} PCM", self.bits)?;
        if !self.packed {
            f.write_str(" in padded frames")?;
        }
        Ok(())
    }
}

// ffmpeg's QuickTime Vorbis entry keeps the Xiph-laced headers in a `glbl` atom, as Matroska's CodecPrivate does
fn vorbis_track(entry: &[u8], timescale: u32) -> Result<TrackKind, G2gError> {
    let description = SoundDescription::parse(entry, u32::from_le_bytes(*b"msVo"))?;
    let channels = u8::try_from(description.channels)
        .ok()
        .filter(|channels| *channels > 0)
        .ok_or(G2gError::CapsMismatch)?;
    let config = entry
        .get(description.children_at..)
        .and_then(|children| find_box(children, b"glbl"))
        .filter(|laced| vorbis_headers(laced).is_some())
        .ok_or(G2gError::CapsMismatch)?;
    Ok(TrackKind::Audio {
        format: AudioFormat::Vorbis,
        channels,
        sample_rate: timescale,
        config: config.to_vec(),
        stored_pcm: None,
    })
}

// the identification, comment and setup headers of a Xiph-laced Vorbis config, none when it holds anything else
pub(crate) fn vorbis_headers(laced: &[u8]) -> Option<Vec<&[u8]>> {
    const HEADER_TYPES: [u8; 3] = [0x01, 0x03, 0x05];
    let packets = crate::matroska::split_xiph_codec_private(laced)?;
    let well_formed = packets.len() == HEADER_TYPES.len()
        && packets.iter().zip(HEADER_TYPES).all(|(packet, kind)| {
            packet.first() == Some(&kind) && packet[1..].starts_with(b"vorbis")
        });
    well_formed.then_some(packets)
}

// esds objectTypeIndication of MPEG-2 and MPEG-1 audio (MP3) in an `mp4a` entry
const MPEG_AUDIO_OBJECT_TYPES: [u8; 2] = [0x69, 0x6B];

// QuickTime sound description versions 1 and 2 add fields before the child boxes
const SOUND_ENTRY_V0_CHILDREN_AT: usize = 28;
const SOUND_ENTRY_V1_CHILDREN_AT: usize = 44;
const SOUND_ENTRY_V2_CHILDREN_AT: usize = 64;

fn sound_entry_children_at(entry: &[u8]) -> Result<usize, G2gError> {
    let version = entry.get(8..10).ok_or(G2gError::CapsMismatch)?;
    Ok(match version {
        [0, 1] => SOUND_ENTRY_V1_CHILDREN_AT,
        [0, 2] => SOUND_ENTRY_V2_CHILDREN_AT,
        _ => SOUND_ENTRY_V0_CHILDREN_AT,
    })
}

/// Recognize a timed-text sample entry and map it to the [`TrackKind::Text`] the
/// cue payload carries: `tx3g` (3GPP timed text) and `wvtt` (WebVTT-in-MP4) cues
/// de-frame to plain UTF-8, and `stpp` (TTML-in-MP4) samples are TTML documents.
/// An unrecognized text codec declines (the track is skipped, not an error).
fn parse_text_entry(entries: &[u8]) -> Option<TrackKind> {
    if find_box(entries, b"tx3g").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Utf8,
            sample: TextSampleFormat::Tx3g,
        })
    } else if find_box(entries, b"wvtt").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Utf8,
            sample: TextSampleFormat::Wvtt,
        })
    } else if find_box(entries, b"stpp").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Ttml,
            sample: TextSampleFormat::Stpp,
        })
    } else {
        None
    }
}

/// Recognize a closed-caption sample entry (QuickTime `c608` / `c708`, under a
/// `clcp` handler) and map it to the [`TrackKind::ClosedCaption`] carriage. An
/// unrecognized caption codec declines (the track is skipped, not an error).
fn parse_cc_entry(entries: &[u8]) -> Option<TrackKind> {
    if find_box(entries, b"c608").is_some() {
        Some(TrackKind::ClosedCaption {
            format: ClosedCaptionFormat::Cea608,
        })
    } else if find_box(entries, b"c708").is_some() {
        Some(TrackKind::ClosedCaption {
            format: ClosedCaptionFormat::Cea708,
        })
    } else {
        None
    }
}

/// De-frame one raw-caption sample to the `cc_data` triple stream a
/// [`Caps::ClosedCaption`](g2g_core::Caps::ClosedCaption) link carries, so a
/// container caption track and an SEI caption block feed the same decoder.
fn deframe_cc(raw: &[u8], format: ClosedCaptionFormat) -> Vec<u8> {
    match format {
        ClosedCaptionFormat::Cea608 => deframe_c608(raw),
        ClosedCaptionFormat::Cea708 => deframe_c708(raw),
        // Only the two carriages above are ever surfaced as a track, so a future
        // third yields no caption data rather than a guessed framing.
        _ => Vec::new(),
    }
}

/// De-frame a `c608` sample: a sequence of `cdat` (line-21 field 1) / `cdt2`
/// (field 2) atoms whose payloads are raw CEA-608 byte pairs, re-tagged as
/// `cc_data` triples (`cc_type` 0 for field 1, 1 for field 2). Atom sizes are
/// untrusted: a size below the 8-byte header, past the sample end, or an odd
/// payload length stops the scan, and an atom that is neither `cdat` nor `cdt2` is
/// skipped, so a malformed sample yields the pairs read so far rather than
/// panicking.
fn deframe_c608(raw: &[u8]) -> Vec<u8> {
    let mut triples: Vec<CcTriple> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        let cc_type = match &raw[off + 4..off + 8] {
            b"cdat" => Some(0u8),
            b"cdt2" => Some(1u8),
            _ => None,
        };
        let body = &raw[off + 8..off + size];
        if let Some(cc_type) = cc_type {
            if !body.len().is_multiple_of(2) {
                break;
            }
            for pair in body.as_chunks::<2>().0 {
                triples.push(CcTriple {
                    cc_type,
                    b0: pair[0],
                    b1: pair[1],
                });
            }
        }
        off += size;
    }
    write_cc_data(&triples)
}

/// De-frame a `c708` sample: `ccdp` atoms each holding a SMPTE ST 334-2 caption
/// distribution packet, unwrapped to the `cc_data` triples the CDP's ccdata
/// section carries. A CDP that fails to parse (bad identifier, length, or
/// checksum) contributes nothing; atom sizes are untrusted, so a bad one stops the
/// scan.
fn deframe_c708(raw: &[u8]) -> Vec<u8> {
    let mut triples: Vec<CcTriple> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        if &raw[off + 4..off + 8] == b"ccdp" {
            if let Some(mut got) = parse_cdp(&raw[off + 8..off + size]) {
                triples.append(&mut got);
            }
        }
        off += size;
    }
    write_cc_data(&triples)
}

/// Strip a 3GPP timed-text (`tx3g`) sample to its UTF-8 cue bytes: a 2-byte
/// big-endian text length, that many UTF-8 bytes, then optional style / modifier
/// boxes (ignored). A zero-length sample is the gap between cues and forwards as
/// an empty string, so a downstream overlay clears. The length is untrusted: a
/// prefix longer than the sample yields an empty cue rather than panicking.
fn deframe_tx3g(raw: &[u8]) -> Vec<u8> {
    let Some(len) = raw
        .get(0..2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
    else {
        return Vec::new();
    };
    raw.get(2..2usize.saturating_add(len))
        .map(<[u8]>::to_vec)
        .unwrap_or_default()
}

/// Strip a WebVTT-in-MP4 (`wvtt`) sample to its UTF-8 cue text (ISO 14496-30). A
/// sample is a sequence of boxes: each `vttc` (cue) carries a `payl` sub-box with
/// the UTF-8 payload, and a `vtte` (empty cue) carries nothing. The `payl`
/// payloads of every `vttc` in the sample concatenate (newline-separated) to the
/// text shown for the sample's time window; an empty / `vtte`-only sample forwards
/// as an empty string so a downstream overlay clears. Box sizes are untrusted: a
/// size below the 8-byte header or past the sample end stops the scan rather than
/// panicking. `sttg` cue settings (placement) are ignored for now.
fn deframe_wvtt(raw: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        let body = &raw[off + 8..off + size];
        if &raw[off + 4..off + 8] == b"vttc" {
            if let Some(payl) = find_box(body, b"payl") {
                if !out.is_empty() {
                    out.push(b'\n');
                }
                out.extend_from_slice(payl);
            }
        }
        off += size;
    }
    out
}

/// One sample of a `moof`+`mdat` fragment as stored: still framed, and still
/// encrypted when its track is.
#[derive(Debug)]
pub(crate) struct MoofSample<'a> {
    /// The `tfhd` `track_ID`, `None` for a `traf` with no `tfhd`.
    pub(crate) track_id: Option<u32>,
    pub(crate) bytes: &'a [u8],
    pub(crate) timing: FragmentSampleTiming,
    /// `None` for a clear track, or for a sample past the ones the fragment's
    /// crypto describes.
    pub(crate) crypt: Option<SampleCrypt>,
    /// Where the fragment's `moof` sits in the source byte stream, which keys a
    /// rotating content key.
    pub(crate) fragment_at: u64,
}

/// A `moof` waiting for its `mdat`.
struct OpenFragment {
    track_id: Option<u32>,
    timings: Vec<FragmentSampleTiming>,
    crypt: Vec<SampleCrypt>,
    fragment_at: u64,
}

/// Walk the `moof`+`mdat` pairs in `data` and split every sample out of its
/// `mdat`, timed in its track's timescale. `track` maps a fragment's `track_ID`
/// to that track's timescale, encryption and `trex` defaults, and a fragment it returns `None`
/// for is skipped. `base_offset` is where `data` sits in the source byte stream.
///
/// Assumes each `trun`'s samples are stored contiguously from the start of the
/// following `mdat` payload; the `trun` `data_offset` is not honored. This holds
/// for ffmpeg / CMAF output. A non-conforming fragment that positions its sample
/// data elsewhere in the `mdat` is mis-split, not rejected.
// per track, where the next fragment starts when it carries no tfdt, kept across the fragments of one stream
#[derive(Debug, Clone, Default)]
pub(crate) struct FragmentStarts(Vec<(Option<u32>, u64)>);

pub(crate) fn moof_samples<'a, 't>(
    data: &'a [u8],
    base_offset: u64,
    fragment_starts: &mut FragmentStarts,
    track: impl Fn(Option<u32>) -> Option<(u32, Option<&'t CencTrack>, SampleDefaults)>,
) -> Result<Vec<MoofSample<'a>>, G2gError> {
    let mut out = Vec::new();
    // `Some(None)` is an open fragment of a skipped track.
    let mut open: Option<Option<OpenFragment>> = None;
    let fragment_end_ticks = &mut fragment_starts.0;
    for (kind, payload, box_at) in boxes_at(data) {
        match kind {
            b"moof" => {
                let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                let tfhd = find_box(traf, b"tfhd");
                // tfhd: track_ID at payload offset 4 (after version/flags).
                let track_id = tfhd.map(|tfhd| be32(tfhd, 4)).transpose()?;
                let Some((timescale, cenc, trex)) = track(track_id) else {
                    open = Some(None);
                    continue;
                };
                let previous_end = fragment_end_ticks
                    .iter()
                    .position(|(id, _)| *id == track_id);
                let base_time = match find_box(traf, b"tfdt").map(|tfdt| (tfdt.first(), tfdt)) {
                    Some((Some(1), tfdt)) => be64(tfdt, 4)?,
                    Some((Some(0), tfdt)) => be32(tfdt, 4)? as u64,
                    Some(_) => return Err(G2gError::CapsMismatch),
                    None => previous_end.map_or(0, |at| fragment_end_ticks[at].1),
                };
                let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
                // The `tfhd` is where a CMAF fragment declares the sample duration
                // its `trun` then omits.
                let defaults = tfhd
                    .map(tfhd_defaults)
                    .transpose()?
                    .unwrap_or_default()
                    .or_trex(trex);
                let trun_samples = parse_trun(trun, defaults, data.len())?;
                let end = trun_samples.iter().fold(base_time, |end, sample| {
                    end.saturating_add(u64::from(sample.duration))
                });
                match previous_end {
                    Some(at) => fragment_end_ticks[at].1 = end,
                    None => fragment_end_ticks.push((track_id, end)),
                }
                // An encrypted track's per-sample IVs / subsample maps live in the
                // fragment's aux info, addressed from the start of this `moof`.
                let crypt = match cenc {
                    Some(c) => {
                        let from_moof = data.get(box_at..).ok_or(G2gError::CapsMismatch)?;
                        fragment_sample_crypt(traf, from_moof, c, trun_samples.len())?
                    }
                    None => Vec::new(),
                };
                open = Some(Some(OpenFragment {
                    track_id,
                    timings: fragment_sample_timings(base_time, &trun_samples, timescale),
                    crypt,
                    fragment_at: base_offset.saturating_add(box_at as u64),
                }));
            }
            b"mdat" => {
                let Some(fragment) = open.take() else {
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let Some(fragment) = fragment else {
                    continue;
                };
                let mut crypt = fragment.crypt.into_iter();
                let mut at = 0usize;
                for timing in fragment.timings {
                    let end = at
                        .checked_add(timing.size as usize)
                        .ok_or(G2gError::CapsMismatch)?;
                    let bytes = payload.get(at..end).ok_or(G2gError::CapsMismatch)?;
                    at = end;
                    out.push(MoofSample {
                        track_id: fragment.track_id,
                        bytes,
                        timing,
                        crypt: crypt.next(),
                        fragment_at: fragment.fragment_at,
                    });
                }
            }
            _ => {}
        }
    }
    if open.is_some() {
        return Err(G2gError::CapsMismatch); // trailing moof without mdat
    }
    Ok(out)
}

/// A stored sample's clear bytes. A protected sample of an `encrypted` track is
/// decrypted in a copy, one a `seig` group declares clear needs no key at all.
fn clear_sample_bytes<'a>(
    sample: &MoofSample<'a>,
    encrypted: bool,
    decrypt: &mut Option<SampleDecrypt<'_>>,
) -> Result<Cow<'a, [u8]>, G2gError> {
    if !encrypted {
        return Ok(Cow::Borrowed(sample.bytes));
    }
    // Fewer crypto records than samples: fail rather than emit ciphertext.
    let crypt = sample.crypt.as_ref().ok_or(G2gError::CapsMismatch)?;
    if !crypt.protected {
        return Ok(Cow::Borrowed(sample.bytes));
    }
    let decrypt = decrypt.as_deref_mut().ok_or(G2gError::CapsMismatch)?;
    let mut buf = sample.bytes.to_vec();
    decrypt(crypt, sample.fragment_at, &mut buf)?;
    Ok(Cow::Owned(buf))
}

/// Split every sample of a multi-track fMP4 out, keyed by its `track_ID`, for
/// the tracks in `tracks` (fragments of any other track are skipped): video
/// samples are de-framed AVCC->Annex-B with a keyframe scan, audio samples pass
/// through (each is a sync sample).
///
/// An encrypted track's samples are decrypted via `decrypt` before de-framing,
/// using the crypto the fragment's sample auxiliary information and `seig`
/// groups resolve (see [`fragment_sample_crypt`]); an encrypted track with no
/// `decrypt` supplied fails loud (`CapsMismatch`), so a keyless build never
/// emits garbage. `base_offset` is where `data` sits in the source byte stream,
/// so a rotating key can be selected by fragment position.
pub(crate) fn parse_fragments_multi(
    data: &[u8],
    tracks: &[TrackHeader],
    base_offset: u64,
    fragment_starts: &mut FragmentStarts,
    mut decrypt: Option<SampleDecrypt<'_>>,
) -> Result<Vec<(u32, Sample)>, G2gError> {
    let track_of = |track_id: Option<u32>| tracks.iter().find(|t| Some(t.track_id) == track_id);
    let stored = moof_samples(data, base_offset, fragment_starts, |track_id| {
        track_of(track_id).map(|t| (t.timescale, t.cenc.as_ref(), t.trex_defaults))
    })?;
    let mut out = Vec::with_capacity(stored.len());
    for sample in &stored {
        let track = track_of(sample.track_id).expect("the walk kept only known tracks");
        let bytes = clear_sample_bytes(sample, track.cenc.is_some(), &mut decrypt)?;
        let (annexb, keyframe) = match &track.kind {
            TrackKind::Video { codec, .. } => {
                let annexb = if length_prefixed(*codec) {
                    avcc_to_annexb(&bytes)?
                } else {
                    bytes.into_owned()
                };
                let keyframe = fragment_video_keyframe(*codec, sample.timing.sync, &annexb);
                (annexb, keyframe)
            }
            TrackKind::RawVideo {
                format,
                width,
                height,
                storage,
            } => {
                let framing = RawVideoFraming {
                    format: *format,
                    storage: *storage,
                    width: *width,
                    height: *height,
                };
                (raw_video_sample(&bytes, framing), true)
            }
            // Audio access units are stored verbatim; each is a sync point.
            TrackKind::Audio {
                format,
                stored_pcm: Some(stored),
                ..
            } => {
                let mut samples = bytes.into_owned();
                stored.to_graph(&mut samples, *format);
                (samples, true)
            }
            TrackKind::Audio { .. } => (bytes.into_owned(), true),
            // A caption sample de-frames to the cc_data triple stream.
            TrackKind::ClosedCaption { format } => (deframe_cc(&bytes, *format), true),
            // A timed-text cue is independent; strip its sample framing
            // (tx3g / wvtt) or pass the whole TTML document (stpp).
            TrackKind::Text { sample, .. } => {
                let cue = match SampleFraming::of_text(*sample) {
                    SampleFraming::WvttText => deframe_wvtt(&bytes),
                    SampleFraming::PassThrough => bytes.into_owned(),
                    _ => deframe_tx3g(&bytes),
                };
                (cue, true)
            }
        };
        out.push((
            track.track_id,
            Sample {
                annexb,
                pts_ns: sample.timing.pts_ns,
                dts_ns: sample.timing.dts_ns,
                duration_ns: sample.timing.duration_ns,
                keyframe,
            },
        ));
    }
    Ok(out)
}

// libavformat keys every sample of an intra-only codec whatever its flags say
fn fragment_video_keyframe(codec: VideoCodec, sync: Option<bool>, au: &[u8]) -> bool {
    match sync {
        Some(sync) => sync || intra_only(codec),
        None => au_is_keyframe(codec, au),
    }
}

// `parse_hvcc` moved to `annexb` (shared with the no_std Matroska demuxer).
pub(crate) use crate::annexb::parse_hvcc;

// The avcC / parameter-set helpers moved to the ungated `annexb` module (M662,
// the no_std FLV demuxer shares them); re-exported so this module's users keep
// their import path.
pub(crate) use crate::annexb::{parse_avcc, prepend_param_sets, starts_with_param_set};

/// Split every sample of a single-track fMP4 out of its `mdat`, converting AVCC
/// framing back to Annex-B. `codec` selects the IDR NAL type used to flag
/// keyframes (the seek snap points). Every fragment is taken as this track's,
/// whatever its `track_ID`.
pub(crate) fn parse_fragments(
    data: &[u8],
    timescale: u32,
    codec: VideoCodec,
    cenc: Option<&CencTrack>,
    base_offset: u64,
    fragment_starts: &mut FragmentStarts,
    mut decrypt: Option<SampleDecrypt<'_>>,
) -> Result<Vec<Sample>, G2gError> {
    let stored = moof_samples(data, base_offset, fragment_starts, |_| {
        Some((timescale, cenc, SampleDefaults::default()))
    })?;
    let mut samples = Vec::with_capacity(stored.len());
    for sample in &stored {
        let bytes = clear_sample_bytes(sample, cenc.is_some(), &mut decrypt)?;
        let annexb = if length_prefixed(codec) {
            avcc_to_annexb(&bytes)?
        } else {
            bytes.into_owned()
        };
        let keyframe = fragment_video_keyframe(codec, sample.timing.sync, &annexb);
        samples.push(Sample {
            annexb,
            pts_ns: sample.timing.pts_ns,
            dts_ns: sample.timing.dts_ns,
            duration_ns: sample.timing.duration_ns,
            keyframe,
        });
    }
    Ok(samples)
}

/// Parse a *progressive* (non-fragmented) MP4: the classic `ftyp/moov/mdat`
/// layout where the `moov`'s sample tables (`stbl`) describe every sample's size
/// (`stsz`), decode duration (`stts`), composition offset (`ctts`), sync flag
/// (`stss`), and chunk layout (`stsc` + `stco`/`co64`), with the elementary data
/// sitting in `mdat` addressed by absolute file offset. This is what most tools
/// write by default (what `Mp4Src` falls back to when a file has no `moof`).
/// Returns the samples in decode order as Annex-B, the same shape
/// [`parse_fragments`] yields, so `Mp4Src::run` is identical downstream.
///
/// Single video track (the first `trak`, matching [`parse_header`]); the absolute
/// chunk offsets are read straight from `data`, so the `mdat` box framing (and
/// any 64-bit `largesize`) never matters.
pub(crate) fn parse_progressive(data: &[u8], timescale: u32) -> Result<Vec<Sample>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    // Single video track. H.264/H.265 samples are AVCC length-prefixed and
    // de-frame to Annex-B; MPEG-4 Part 2 (`mp4v`) samples are already raw
    // elementary stream (start codes) and AV1 (`av01`) samples plain
    // low-overhead OBUs, so those pass through verbatim. Absent an stsd,
    // default to Annex-B de-framing (the H.264/H.265 case).
    let framing = find_path(mdia, &[b"minf", b"stbl", b"stsd"])
        .and_then(|stsd| stsd.get(8..))
        .filter(|entries| {
            find_box(entries, b"mp4v").is_some() || find_box(entries, b"av01").is_some()
        })
        .map_or(SampleFraming::Video, |_| SampleFraming::PassThrough);
    parse_progressive_track(data, stbl, timescale, framing)
}

/// How a progressive track's stored samples map to the elementary bytes the
/// pipeline carries: video de-frames AVCC -> Annex-B, audio passes through
/// verbatim (each is a sync sample), and a `tx3g` timed-text sample strips its
/// 2-byte length prefix to the UTF-8 cue.
#[derive(Debug, Clone, Copy)]
enum SampleFraming {
    Video,
    PassThrough,
    // a long-tail video track, which also keeps a sample cut short by the end of a truncated file
    TaggedVideo,
    Tx3gText,
    WvttText,
    ClosedCaption(ClosedCaptionFormat),
    Pcm(PcmFraming),
    RawVideo(RawVideoFraming),
    // a coded QuickTime sound track, which an stts of unit durations stores by chunk
    #[cfg(feature = "ffmpeg-long-tail")]
    QuickTimeChunks(SoundDescription),
}

// a QuickTime PCM track stores one frame per sample, whatever size its stsz gives
#[derive(Debug, Clone, Copy)]
struct PcmFraming {
    format: AudioFormat,
    frame_bytes: usize,
    stored: StoredPcm,
}

impl SampleFraming {
    /// The de-framing a track's [`TrackKind`] selects.
    fn of(kind: &TrackKind) -> Self {
        match kind {
            TrackKind::Video { codec, .. } if length_prefixed(*codec) => SampleFraming::Video,
            TrackKind::Video {
                codec: VideoCodec::Tagged(_),
                ..
            } => SampleFraming::TaggedVideo,
            TrackKind::Video { .. } => SampleFraming::PassThrough,
            TrackKind::RawVideo {
                format,
                width,
                height,
                storage,
            } => SampleFraming::RawVideo(RawVideoFraming {
                format: *format,
                storage: *storage,
                width: *width,
                height: *height,
            }),
            TrackKind::Audio {
                format,
                channels,
                stored_pcm: Some(stored),
                ..
            } => match format.bytes_per_sample() {
                Some(width) => SampleFraming::Pcm(PcmFraming {
                    format: *format,
                    frame_bytes: width * usize::from(*channels),
                    stored: *stored,
                }),
                None => SampleFraming::PassThrough,
            },
            #[cfg(feature = "ffmpeg-long-tail")]
            TrackKind::Audio {
                format: AudioFormat::Tagged(tag),
                config,
                ..
            } => match config
                .get(crate::mp4box::BOX_HEADER_LEN..)
                .map(|payload| SoundDescription::parse(payload, tag.tag))
            {
                Some(Ok(sound)) => SampleFraming::QuickTimeChunks(sound),
                _ => SampleFraming::PassThrough,
            },
            TrackKind::Audio { .. } => SampleFraming::PassThrough,
            TrackKind::Text { sample, .. } => Self::of_text(*sample),
            TrackKind::ClosedCaption { format } => SampleFraming::ClosedCaption(*format),
        }
    }

    /// The de-framing a timed-text sample format selects: a `stpp` sample is the
    /// whole TTML document (pass through), `tx3g` / `wvtt` strip their framing.
    fn of_text(sample: TextSampleFormat) -> Self {
        match sample {
            TextSampleFormat::Tx3g => SampleFraming::Tx3gText,
            TextSampleFormat::Wvtt => SampleFraming::WvttText,
            TextSampleFormat::Stpp => SampleFraming::PassThrough,
        }
    }
}

/// Parse the samples of one progressive track from its `stbl` sample tables,
/// addressing the elementary data in `data` by the absolute chunk offsets. The
/// per-track core shared by [`parse_progressive`] (single video track) and
/// [`parse_progressive_multi`] (every A/V / text track). `framing` selects how a
/// stored sample maps to the elementary bytes (AVCC->Annex-B, pass-through, or
/// tx3g text de-framing).
fn parse_progressive_track(
    data: &[u8],
    stbl: &[u8],
    timescale: u32,
    framing: SampleFraming,
) -> Result<Vec<Sample>, G2gError> {
    // stsz: per-sample sizes. A non-zero `default_size` means every sample is
    // that size (no table); otherwise a `sample_count`-long table follows.
    let stsz = find_box(stbl, b"stsz").ok_or(G2gError::CapsMismatch)?;
    let default_size = be32(stsz, 4)?;
    // stsz counts PCM samples here, many to a stored byte, so the per-sample walk below does not apply
    #[cfg(feature = "ffmpeg-long-tail")]
    if let SampleFraming::QuickTimeChunks(sound) = framing {
        let stts = find_box(stbl, b"stts").ok_or(G2gError::CapsMismatch)?;
        if be32(stts, 4)? == 1 && be32(stts, 12)? == 1 {
            let chunks = chunk_sample_counts(stbl, &chunk_offsets(stbl)?)?;
            return Ok(quicktime_chunk_samples(
                data,
                &chunks,
                timescale,
                sound,
                default_size,
            ));
        }
    }
    // A sample needs at least one byte of media data, so no sample past the
    // file size is ever read, whether the stsz lies or the file was cut short.
    // Clamp before the per-sample allocations below (the default_size branch
    // fills the Vec, committing physical pages).
    let sample_count = (be32(stsz, 8)? as usize).min(data.len());
    let sizes: Vec<u32> = if let SampleFraming::Pcm(pcm) = framing {
        let frame_bytes = u32::try_from(pcm.frame_bytes).map_err(|_| G2gError::CapsMismatch)?;
        alloc::vec![frame_bytes; sample_count]
    } else if default_size != 0 {
        alloc::vec![default_size; sample_count]
    } else {
        (0..sample_count)
            .map(|i| be32(stsz, 12 + i * 4))
            .collect::<Result<_, _>>()?
    };

    // stts: decode durations, run-length encoded, expanded to one per sample.
    let stts = find_box(stbl, b"stts").ok_or(G2gError::CapsMismatch)?;
    let mut durations: Vec<u32> = Vec::with_capacity(sample_count);
    for e in 0..be32(stts, 4)? as usize {
        let cnt = be32(stts, 8 + e * 8)? as usize;
        let delta = be32(stts, 12 + e * 8)?;
        durations.resize(durations.len().saturating_add(cnt).min(sample_count), delta);
    }
    durations.resize(sample_count, 0);

    let ctts_offsets = composition_offsets(stbl, sample_count)?;

    let chunk_offsets = chunk_offsets(stbl)?;

    // stsc: how many samples sit in each chunk, run-length over chunk ranges.
    // Resolve to a samples-per-chunk count for every chunk.
    let stsc = find_box(stbl, b"stsc").ok_or(G2gError::CapsMismatch)?;
    let stsc_n = be32(stsc, 4)? as usize;
    if stsc_n == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let stsc_entry = |i: usize| -> Result<(u32, u32, u32), G2gError> {
        Ok((
            be32(stsc, 8 + i * 12)?,
            be32(stsc, 12 + i * 12)?,
            be32(stsc, 16 + i * 12)?,
        ))
    };
    // Place each sample at its file offset: within a chunk samples are
    // contiguous, so offset advances by the running sample size.
    let mut sample_offsets: Vec<u64> = Vec::with_capacity(sample_count);
    let mut sample_descriptions: Vec<u32> = Vec::with_capacity(sample_count);
    // (file offset, sample count) of each chunk
    let mut chunks: Vec<(u64, usize)> = Vec::with_capacity(chunk_offsets.len());
    let mut si = 0usize;
    // Chunks ascend, so the stsc cursor only moves forward.
    let mut next_entry = 0usize;
    let mut spc = 0u32;
    let mut description = FIRST_SAMPLE_DESCRIPTION;
    'chunks: for (ci, &chunk_off) in chunk_offsets.iter().enumerate() {
        // The samples-per-chunk for this chunk is the last stsc entry whose
        // (1-based) first_chunk does not exceed it.
        let chunk_1based = (ci + 1) as u32;
        while next_entry < stsc_n {
            let (first_chunk, samples_per_chunk, sample_description) = stsc_entry(next_entry)?;
            if first_chunk > chunk_1based {
                break;
            }
            spc = samples_per_chunk;
            description = sample_description;
            next_entry += 1;
        }
        let mut at = chunk_off;
        chunks.push((chunk_off, 0));
        for _ in 0..spc {
            if si >= sample_count {
                break 'chunks;
            }
            sample_offsets.push(at);
            sample_descriptions.push(description);
            at = at.saturating_add(sizes[si] as u64);
            si += 1;
            if let Some((_, count)) = chunks.last_mut() {
                *count += 1;
            }
        }
    }
    if sample_offsets.len() != sample_count {
        return Err(G2gError::CapsMismatch); // stsc/stco disagree with stsz
    }
    if let SampleFraming::Pcm(pcm) = framing {
        return Ok(pcm_chunk_samples(data, &chunks, &durations, timescale, pcm));
    }

    // stss: 1-based sync-sample numbers (ascending). Absent => every sample is a
    // sync sample (e.g. all-intra). Used as the keyframe flag (seek snap points).
    // Short-circuit on the first out-of-range entry (like stco/stsz) so a bogus
    // count fails loud instead of spinning the full untrusted range.
    let sync: Option<Vec<u32>> = match find_box(stbl, b"stss") {
        Some(stss) => Some(
            (0..be32(stss, 4)? as usize)
                .map(|i| be32(stss, 8 + i * 4))
                .collect::<Result<_, _>>()?,
        ),
        None => None,
    };

    // a sample description switch carries its own parameter sets, so they go in-band
    let description_param_sets = match framing {
        SampleFraming::Video => video_entry_param_sets(stbl)?,
        _ => Vec::new(),
    };
    let mut previous_description = FIRST_SAMPLE_DESCRIPTION;
    let mut samples = Vec::with_capacity(sample_count);
    let mut dts: u64 = 0;
    for i in 0..sample_count {
        let off = sample_offsets[i] as usize;
        // `off` comes from an untrusted co64/stco chunk offset, so bound the end
        // with checked arithmetic (a u64 offset near usize::MAX would otherwise
        // overflow the `off + size` add and panic in debug).
        let end = off
            .checked_add(sizes[i] as usize)
            .ok_or(G2gError::CapsMismatch)?;
        // a truncated file ends the track at its last whole sample, except that
        // libavcodec also decodes what remains of a cut long-tail video sample
        let (raw, cut) = match data.get(off..end) {
            Some(raw) => (raw, false),
            None => match data.get(off..) {
                Some(rest) if !rest.is_empty() && matches!(framing, SampleFraming::TaggedVideo) => {
                    (rest, true)
                }
                _ => break,
            },
        };
        let pts = (dts as i64).saturating_add(ctts_offsets[i]).max(0) as u64;
        let keyframe = match &sync {
            Some(list) => list.binary_search(&((i + 1) as u32)).is_ok(),
            None => true,
        };
        let annexb = match framing {
            SampleFraming::Video => {
                let annexb = avcc_to_annexb(raw)?;
                let description = sample_descriptions[i];
                let switched = description != previous_description;
                previous_description = description;
                match description_param_sets.get(description.wrapping_sub(1) as usize) {
                    Some(Some((codec, sets))) if switched => {
                        prepend_param_sets(&annexb, sets, *codec)
                    }
                    _ => annexb,
                }
            }
            SampleFraming::PassThrough | SampleFraming::TaggedVideo => raw.to_vec(),
            SampleFraming::Tx3gText => deframe_tx3g(raw),
            SampleFraming::WvttText => deframe_wvtt(raw),
            SampleFraming::ClosedCaption(format) => deframe_cc(raw, format),
            SampleFraming::RawVideo(framing) => raw_video_sample(raw, framing),
            SampleFraming::Pcm(_) => unreachable!("PCM tracks return per chunk above"),
            #[cfg(feature = "ffmpeg-long-tail")]
            SampleFraming::QuickTimeChunks(_) => raw.to_vec(),
        };
        samples.push(Sample {
            annexb,
            pts_ns: ticks_to_ns(pts, timescale),
            dts_ns: ticks_to_ns(dts, timescale),
            duration_ns: ticks_to_ns(durations[i] as u64, timescale),
            keyframe,
        });
        if cut {
            break;
        }
        dts = dts.saturating_add(durations[i] as u64);
    }
    Ok(samples)
}

// one sample per chunk of PCM frames, cut at the last whole frame of a truncated file
fn pcm_chunk_samples(
    data: &[u8],
    chunks: &[(u64, usize)],
    durations: &[u32],
    timescale: u32,
    pcm: PcmFraming,
) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(chunks.len());
    let mut first_frame = 0usize;
    let mut dts: u64 = 0;
    for &(offset, frames) in chunks {
        if frames == 0 {
            continue;
        }
        let stored = usize::try_from(offset)
            .ok()
            .and_then(|offset| data.get(offset..))
            .unwrap_or_default();
        let whole_frames = frames.min(stored.len() / pcm.frame_bytes);
        if whole_frames == 0 {
            break;
        }
        let mut bytes = stored[..whole_frames * pcm.frame_bytes].to_vec();
        pcm.stored.to_graph(&mut bytes, pcm.format);
        let duration: u64 = durations[first_frame..first_frame + whole_frames]
            .iter()
            .map(|&ticks| u64::from(ticks))
            .sum();
        samples.push(Sample {
            annexb: bytes,
            pts_ns: ticks_to_ns(dts, timescale),
            dts_ns: ticks_to_ns(dts, timescale),
            duration_ns: ticks_to_ns(duration, timescale),
            keyframe: true,
        });
        if whole_frames < frames {
            break;
        }
        first_frame += frames;
        dts = dts.saturating_add(duration);
    }
    samples
}

// stco (32-bit) or co64 (64-bit): per-chunk file offsets
fn chunk_offsets(stbl: &[u8]) -> Result<Vec<u64>, G2gError> {
    if let Some(stco) = find_box(stbl, b"stco") {
        return (0..be32(stco, 4)? as usize)
            .map(|c| be32(stco, 8 + c * 4).map(u64::from))
            .collect();
    }
    let co64 = find_box(stbl, b"co64").ok_or(G2gError::CapsMismatch)?;
    (0..be32(co64, 4)? as usize)
        .map(|c| be64(co64, 8 + c * 8))
        .collect()
}

// each chunk's file offset and the samples stsc puts in it
#[cfg(feature = "ffmpeg-long-tail")]
fn chunk_sample_counts(stbl: &[u8], offsets: &[u64]) -> Result<Vec<(u64, usize)>, G2gError> {
    const STSC_ENTRY_LEN: usize = 12;
    let stsc = find_box(stbl, b"stsc").ok_or(G2gError::CapsMismatch)?;
    let entries = be32(stsc, 4)? as usize;
    let mut next_entry = 0;
    let mut samples_per_chunk = 0;
    let mut chunks = Vec::with_capacity(offsets.len());
    for (index, &offset) in offsets.iter().enumerate() {
        // stsc numbers chunks from 1, and an entry runs until the next one's first chunk
        while next_entry < entries
            && be32(stsc, 8 + next_entry * STSC_ENTRY_LEN)? as usize <= index + 1
        {
            samples_per_chunk = be32(stsc, 12 + next_entry * STSC_ENTRY_LEN)? as usize;
            next_entry += 1;
        }
        chunks.push((offset, samples_per_chunk));
    }
    Ok(chunks)
}

// One sample per chunk, its bytes what mov.c reads for the chunk's sample count:
// whole frames of the sound description, or stsz's size per sample.
#[cfg(feature = "ffmpeg-long-tail")]
fn quicktime_chunk_samples(
    data: &[u8],
    chunks: &[(u64, usize)],
    timescale: u32,
    sound: SoundDescription,
    sample_size: u32,
) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(chunks.len());
    let mut dts: u64 = 0;
    for &(offset, count) in chunks.iter().filter(|(_, count)| *count > 0) {
        let count = count as u64;
        let len = if sound.samples_per_frame > 1 {
            count / u64::from(sound.samples_per_frame) * u64::from(sound.bytes_per_frame)
        } else {
            count.saturating_mul(u64::from(sample_size))
        };
        let bytes = usize::try_from(offset)
            .ok()
            .zip(usize::try_from(len).ok())
            .and_then(|(start, len)| data.get(start..start.checked_add(len)?));
        // a truncated file ends the track at its last whole chunk
        let Some(bytes) = bytes.filter(|bytes| !bytes.is_empty()) else {
            break;
        };
        samples.push(Sample {
            annexb: bytes.to_vec(),
            pts_ns: ticks_to_ns(dts, timescale),
            dts_ns: ticks_to_ns(dts, timescale),
            duration_ns: ticks_to_ns(count, timescale),
            keyframe: true,
        });
        dts = dts.saturating_add(count);
    }
    samples
}

// stsc numbers sample descriptions from 1, and the demuxers prepend the first one's sets
const FIRST_SAMPLE_DESCRIPTION: u32 = 1;

// one sample entry's codec and config-record parameter sets
type EntryParamSets = (VideoCodec, Vec<Vec<u8>>);

// encoders write negative offsets into version 0 too, so both versions read signed, as in ffmpeg
fn composition_offsets(stbl: &[u8], sample_count: usize) -> Result<Vec<i64>, G2gError> {
    let Some(ctts) = find_box(stbl, b"ctts") else {
        return Ok(alloc::vec![0i64; sample_count]);
    };
    let mut out: Vec<i64> = Vec::with_capacity(sample_count);
    for e in 0..be32(ctts, 4)? as usize {
        let cnt = be32(ctts, 8 + e * 8)? as usize;
        let off = i64::from(be32(ctts, 12 + e * 8)? as i32);
        let target = out.len().saturating_add(cnt).min(sample_count);
        out.resize(target, off);
    }
    out.resize(sample_count, 0);
    Ok(out)
}

// the parameter sets of each H.264 / H.265 sample entry, in stsd order
fn video_entry_param_sets(stbl: &[u8]) -> Result<Vec<Option<EntryParamSets>>, G2gError> {
    let Some(entries) = find_box(stbl, b"stsd").and_then(|stsd| stsd.get(8..)) else {
        return Ok(Vec::new());
    };
    boxes(entries)
        .map(|(kind, entry)| {
            let children = entry.get(78..).unwrap_or_default();
            Ok(match kind {
                b"avc1" | b"avc3" => match find_box(children, b"avcC") {
                    Some(avcc) => Some((VideoCodec::H264, parse_avcc(avcc)?.parameter_sets)),
                    None => None,
                },
                b"hvc1" | b"hev1" => match find_box(children, b"hvcC") {
                    Some(hvcc) => Some((VideoCodec::H265, parse_hvcc(hvcc)?.parameter_sets)),
                    None => None,
                },
                _ => None,
            })
        })
        .collect()
}

/// Parse a progressive (`moov`+`mdat`, no `moof`) multi-track file: every A/V /
/// text track's samples, keyed by `track_ID`, interleaved in decode-time order. The progressive analog
/// of [`parse_fragments_multi`] (and the multi-track form of [`parse_progressive`]),
/// for files that carry several tracks in classic sample-table layout rather than
/// fragments. Each track's `stbl` is walked independently against its own
/// timescale and de-framing; tracks absent from `tracks` are skipped.
pub(crate) fn parse_progressive_multi(
    data: &[u8],
    tracks: &[TrackHeader],
) -> Result<Vec<(u32, Sample)>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let mut out = Vec::new();
    for track in tracks {
        let Some(trak) = find_trak_by_id(moov, track.track_id) else {
            continue; // a track with no matching trak box
        };
        let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
        let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
        let framing = SampleFraming::of(&track.kind);
        // an stss listing only some samples does not make an intra-only codec's other samples depend on them
        let every_sample_sync = match &track.kind {
            TrackKind::Video { codec, .. } => intra_only(*codec),
            TrackKind::RawVideo { .. } => true,
            _ => false,
        };
        for mut s in parse_progressive_track(data, stbl, track.timescale, framing)? {
            s.keyframe |= every_sample_sync;
            out.push((track.track_id, s));
        }
    }
    // Stable, so each track keeps its own sample order.
    out.sort_by_key(|(_, sample)| sample.dts_ns);
    Ok(out)
}

/// Cap on the chapters a QuickTime chapter track yields. Its sample count is
/// the file's claim, and each sample costs a whole [`Chapter`], so the walk
/// stops rather than sizing the list from the file. (The `chpl` path needs no
/// cap: its count field is one byte.)
const MAX_CHAPTER_TRACK_CHAPTERS: usize = 4096;

/// The chapters a progressive MP4 declares (M1046): the QuickTime chapter text
/// track when the file has one and `data` holds its samples, else the Nero
/// `chpl` list in `moov/udta`. The text track is preferred because it carries a
/// duration per chapter, which `chpl` cannot express; a `moov`-first (faststart)
/// file being read before its `mdat` lands falls back to `chpl`.
pub(crate) fn parse_chapters(data: &[u8]) -> Vec<Chapter> {
    let Some(moov) = find_box(data, b"moov") else {
        return Vec::new();
    };
    let from_track = chapter_track_chapters(data, moov);
    if !from_track.is_empty() {
        return from_track;
    }
    parse_chpl(moov)
}

/// The chapters of the QuickTime chapter text track, empty when the file has
/// none. A media `trak` points at it with a `tref/chap` naming its `track_ID`;
/// that track's samples are the chapter titles, timed by its own sample table,
/// so one sample is one chapter running for that sample's duration.
fn chapter_track_chapters(data: &[u8], moov: &[u8]) -> Vec<Chapter> {
    let chapter_track_id = boxes(moov)
        .filter(|(kind, _)| *kind == b"trak")
        // A `tref/chap` may name several tracks; the first is the chapter track.
        .find_map(|(_, trak)| be32(find_path(trak, &[b"tref", b"chap"])?, 0).ok());
    let Some(chapter_track_id) = chapter_track_id else {
        return Vec::new();
    };
    let Some(trak) = find_trak_by_id(moov, chapter_track_id) else {
        return Vec::new();
    };
    let Some(mdia) = find_box(trak, b"mdia") else {
        return Vec::new();
    };
    // mdhd v0: timescale at payload offset 12, like every other track here.
    let Some(mdhd) = find_box(mdia, b"mdhd").filter(|m| m.first() == Some(&0)) else {
        return Vec::new();
    };
    let Ok(timescale) = be32(mdhd, 12) else {
        return Vec::new();
    };
    if timescale == 0 {
        return Vec::new();
    }
    let Some(stbl) = find_path(mdia, &[b"minf", b"stbl"]) else {
        return Vec::new();
    };
    let Ok(samples) = parse_progressive_track(data, stbl, timescale, SampleFraming::PassThrough)
    else {
        return Vec::new();
    };
    samples
        .iter()
        .take(MAX_CHAPTER_TRACK_CHAPTERS)
        .filter_map(|sample| {
            let title = chapter_sample_title(&sample.annexb)?;
            Some(Chapter {
                start_ns: sample.pts_ns,
                end_ns: Some(sample.pts_ns.saturating_add(sample.duration_ns)),
                title,
                language: None,
                sub_chapters: Vec::new(),
            })
        })
        .collect()
}

/// The title out of one QuickTime chapter-track sample: a big-endian `u16` byte
/// length, the text, then trailing atoms (an `encd` text encoding) this ignores.
/// The text is UTF-8 unless it opens with a byte-order mark, which is how
/// QuickTime and iTunes store a non-ASCII title.
fn chapter_sample_title(sample: &[u8]) -> Option<String> {
    let length = u16::from_be_bytes(sample.get(0..2)?.try_into().ok()?) as usize;
    let text = sample.get(2..2 + length)?;
    match text {
        [0xFE, 0xFF, rest @ ..] => decode_utf16(rest, true),
        [0xFF, 0xFE, rest @ ..] => decode_utf16(rest, false),
        _ => core::str::from_utf8(text).ok().map(String::from),
    }
}

/// Decode UTF-16 code units of the given endianness, or `None` for an odd byte
/// count or an unpaired surrogate.
fn decode_utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units = bytes.as_chunks::<2>().0.iter().map(|pair| {
        let pair = [pair[0], pair[1]];
        if big_endian {
            u16::from_be_bytes(pair)
        } else {
            u16::from_le_bytes(pair)
        }
    });
    char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .ok()
}

/// The `trak` box (payload) in `moov` whose `tkhd` carries `track_id`, or `None`.
fn find_trak_by_id(moov: &[u8], track_id: u32) -> Option<&[u8]> {
    boxes(moov)
        .filter(|(k, _)| *k == b"trak")
        .map(|(_, t)| t)
        .find(|trak| {
            find_box(trak, b"tkhd")
                .and_then(|tkhd| read_tkhd(tkhd).ok())
                .map(|tkhd| tkhd.track_id)
                == Some(track_id)
        })
}

// the sample duration and size a tfhd or trex declares, each 0 when undeclared
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SampleDefaults {
    pub(crate) duration: u32,
    pub(crate) size: u32,
    pub(crate) flags: Option<u32>,
}

impl SampleDefaults {
    // a field the tfhd declares overrides the trex one
    fn or_trex(self, trex: SampleDefaults) -> SampleDefaults {
        SampleDefaults {
            duration: if self.duration == 0 {
                trex.duration
            } else {
                self.duration
            },
            size: if self.size == 0 { trex.size } else { self.size },
            flags: self.flags.or(trex.flags),
        }
    }
}

/// A `traf`'s `tfhd` `default_sample_duration`, `default_sample_size` and
/// `default_sample_flags` (ISO 14496-12 8.8.7), each absent when the box does not
/// declare it. A `trun` may
/// lean on any of them instead of repeating it per sample, which is what ffmpeg's
/// fragmented and CMAF output does (a constant-bitrate audio fragment often
/// carries neither field in its `trun`).
pub(crate) fn tfhd_defaults(tfhd: &[u8]) -> Result<SampleDefaults, G2gError> {
    let flags = be32(tfhd, 0)? & 0x00FF_FFFF;
    // version/flags + track_ID, then the optional fields in declaration order.
    let mut at = 8usize;
    if flags & 0x01 != 0 {
        at += 8; // base_data_offset
    }
    if flags & 0x02 != 0 {
        at += 4; // sample_description_index
    }
    let duration = if flags & 0x08 != 0 {
        let d = be32(tfhd, at)?;
        at += 4;
        d
    } else {
        0
    };
    let size = if flags & 0x10 != 0 {
        let s = be32(tfhd, at)?;
        at += 4;
        s
    } else {
        0
    };
    let sample_flags = if flags & 0x20 != 0 {
        Some(be32(tfhd, at)?)
    } else {
        None
    };
    Ok(SampleDefaults {
        duration,
        size,
        flags: sample_flags,
    })
}

/// `sample_flags` bit 16: the sample is *not* a sync sample (ISO/IEC 14496-12).
const NON_SYNC: u32 = 0x0001_0000;
// sample_depends_on 1: the sample references other samples
const DEPENDS_ON_OTHERS: u32 = 0x0100_0000;

// libavformat reads either bit as a non-keyframe
fn sample_flags_sync(flags: u32) -> bool {
    flags & (NON_SYNC | DEPENDS_ON_OTHERS) == 0
}

/// Whether the first sample a `trun` describes is a sync sample, i.e. whether
/// the fragment it belongs to opens at a random access point. A `trun` that
/// states no flags at all (neither the first-sample override nor per-sample
/// flags) is taken to open at one, which is what a fragment without sample
/// dependency information implies.
pub(crate) fn trun_first_sample_is_sync(trun: &[u8]) -> Result<bool, G2gError> {
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    // version/flags + sample_count, then the optional fields in declaration order.
    let mut at = 8usize;
    if flags & 0x001 != 0 {
        at += 4; // data_offset
    }
    if flags & 0x004 != 0 {
        return Ok(be32(trun, at)? & NON_SYNC == 0); // first_sample_flags
    }
    if flags & 0x400 != 0 {
        if flags & 0x100 != 0 {
            at += 4; // sample_duration
        }
        if flags & 0x200 != 0 {
            at += 4; // sample_size
        }
        return Ok(be32(trun, at)? & NON_SYNC == 0);
    }
    Ok(true)
}

// durations and offsets in track timescale ticks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrunSample {
    pub(crate) size: u32,
    pub(crate) duration: u32,
    pub(crate) composition_offset: i64,
    // None when neither the trun nor the defaults carry sample flags
    pub(crate) sync: Option<bool>,
}

/// `trun` (v0 or v1). A `trun` that omits the per-sample duration, size or
/// flags takes them from `defaults`; a duration neither
/// carries is `0`, but a *size* neither carries is fatal (the samples could not
/// be split). Both versions read the per-sample composition-time offset signed,
/// as ffmpeg does, since encoders write negative offsets into v0 too.
pub(crate) fn parse_trun(
    trun: &[u8],
    defaults: SampleDefaults,
    fragment_len: usize,
) -> Result<Vec<TrunSample>, G2gError> {
    if !matches!(trun.first(), Some(0 | 1)) {
        return Err(G2gError::CapsMismatch); // unknown trun version
    }
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    if flags & 0x200 == 0 && defaults.size == 0 {
        return Err(G2gError::CapsMismatch); // no sample sizes anywhere
    }
    let count = be32(trun, 4)? as usize;
    let mut at = 8usize;
    if flags & 0x1 != 0 {
        at += 4; // data offset (sequential mdat split makes it redundant)
    }
    let mut first_sample_flags = defaults.flags;
    if flags & 0x4 != 0 {
        first_sample_flags = Some(be32(trun, at)?);
        at += 4;
    }
    // Each sample consumes at least its 4-byte size plus the optional per-sample
    // fields, so an untrusted `count` cannot exceed the bytes that remain. Reject
    // a lying count before reserving capacity for it.
    let per_sample = if flags & 0x200 != 0 { 4 } else { 0 }
        + if flags & 0x100 != 0 { 4 } else { 0 }
        + if flags & 0x400 != 0 { 4 } else { 0 }
        + if flags & 0x800 != 0 { 4 } else { 0 };
    // Every field defaulted from the `tfhd` leaves the count unconstrained by the
    // `trun`, so bound it by the fragment instead (each sample occupies bytes
    // there); otherwise by the per-sample records the box can hold.
    let bound = trun
        .len()
        .saturating_sub(at)
        .checked_div(per_sample)
        .unwrap_or(fragment_len);
    if count > bound {
        return Err(G2gError::CapsMismatch);
    }
    let mut samples = Vec::with_capacity(count);
    for index in 0..count {
        let mut duration = defaults.duration;
        if flags & 0x100 != 0 {
            duration = be32(trun, at)?;
            at += 4;
        }
        let mut size = defaults.size;
        if flags & 0x200 != 0 {
            size = be32(trun, at)?;
            at += 4;
        }
        let mut sample_flags = if index == 0 {
            first_sample_flags
        } else {
            defaults.flags
        };
        if flags & 0x400 != 0 {
            sample_flags = Some(be32(trun, at)?);
            at += 4;
        }
        let mut composition_offset = 0;
        if flags & 0x800 != 0 {
            composition_offset = i64::from(be32(trun, at)? as i32);
            at += 4;
        }
        samples.push(TrunSample {
            size,
            duration,
            composition_offset,
            sync: sample_flags.map(sample_flags_sync),
        });
    }
    Ok(samples)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FragmentSampleTiming {
    pub(crate) size: u32,
    pub(crate) sync: Option<bool>,
    pub(crate) dts_ns: u64,
    pub(crate) pts_ns: u64,
    pub(crate) duration_ns: u64,
}

fn fragment_sample_timings(
    base_time: u64,
    samples: &[TrunSample],
    timescale: u32,
) -> Vec<FragmentSampleTiming> {
    let mut decode_time = base_time;
    samples
        .iter()
        .map(|sample| {
            let timing = FragmentSampleTiming {
                size: sample.size,
                sync: sample.sync,
                dts_ns: ticks_to_ns(decode_time, timescale),
                pts_ns: ticks_to_ns(
                    decode_time.saturating_add_signed(sample.composition_offset),
                    timescale,
                ),
                duration_ns: ticks_to_ns(u64::from(sample.duration), timescale),
            };
            // base_time and durations are untrusted
            decode_time = decode_time.saturating_add(u64::from(sample.duration));
            timing
        })
        .collect()
}

// H.264 / H.265 samples are AVCC, every other codec stores its elementary bytes as is
fn length_prefixed(codec: VideoCodec) -> bool {
    matches!(codec, VideoCodec::H264 | VideoCodec::H265)
}

/// 4-byte-length-prefixed AVCC NALUs back to Annex-B start codes.
fn avcc_to_annexb(avcc: &[u8]) -> Result<Vec<u8>, G2gError> {
    crate::annexb::length_prefixed_to_annexb(avcc, crate::annexb::AVCC_LENGTH_SIZE)
        .ok_or(G2gError::CapsMismatch)
}

/// Incremental CMAF chunk splitter for a segment still being written (M888): fed
/// a segment response as its bytes arrive, it yields each complete chunk (the run
/// of boxes up to and including an `mdat`), so a low-latency packager's
/// `styp`+`moof`+`mdat` chunks flow downstream while the rest of the segment is
/// still on the wire. Every fed byte comes back out exactly once and in order
/// ([`CmafChunker::next_chunk`], then the [`CmafChunker::finish`] remainder), so
/// chunked and whole-response consumption hand a demuxer the same byte stream.
///
/// `max` bounds what may be held for one chunk: a box declaring more than that,
/// or a pending run growing past it, fails loud instead of buffering on an
/// attacker-chosen length.
#[cfg(any(test, feature = "dash"))]
#[derive(Debug)]
pub(crate) struct CmafChunker {
    /// Fed bytes not yet yielded.
    buf: Vec<u8>,
    /// How much of `buf`'s front is already framed as whole boxes (none of them
    /// an `mdat`), so a rescan resumes at the box still arriving.
    framed: usize,
    max: usize,
}

#[cfg(any(test, feature = "dash"))]
impl CmafChunker {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            buf: Vec::new(),
            framed: 0,
            max,
        }
    }

    /// Take newly arrived body bytes.
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        if self.buf.len().saturating_add(bytes.len()) > self.max {
            return Err(G2gError::CapsMismatch);
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// The next complete chunk, or `None` while it is still arriving. Call until
    /// it yields `None` after every [`CmafChunker::feed`].
    pub(crate) fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, G2gError> {
        loop {
            let Some(total) = next_box_len(&self.buf[self.framed..])? else {
                return Ok(None);
            };
            if total > self.max {
                return Err(G2gError::CapsMismatch);
            }
            let end = self.framed.saturating_add(total);
            if self.buf.len() < end {
                return Ok(None); // the rest of this box is still in flight
            }
            let mdat = &self.buf[self.framed + 4..self.framed + 8] == b"mdat";
            self.framed = end;
            if mdat {
                let chunk: Vec<u8> = self.buf.drain(..self.framed).collect();
                self.framed = 0;
                return Ok(Some(chunk));
            }
        }
    }

    /// Whatever is left when the body ends: a chunkless response (an init
    /// segment's `ftyp`+`moov`), a trailing box after the last `mdat`, or the tail
    /// of a truncated transfer. Verbatim, so what reaches the demuxer is the
    /// response body either way.
    pub(crate) fn finish(&mut self) -> Option<Vec<u8>> {
        self.framed = 0;
        (!self.buf.is_empty()).then(|| core::mem::take(&mut self.buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `trak` holding only an `edts/elst` built from raw entry bytes.
    fn trak_with_elst(version: u8, entries: u32, entry: &[u8]) -> Vec<u8> {
        use crate::mp4box::{full_box, mp4_box};
        let mut p = entries.to_be_bytes().to_vec();
        p.extend_from_slice(entry);
        mp4_box(b"edts", &full_box(b"elst", version, 0, &p))
    }

    fn stts_box(entry_count: u32, entries: &[(u32, u32)]) -> Vec<u8> {
        let mut payload = entry_count.to_be_bytes().to_vec();
        for (count, delta) in entries {
            payload.extend_from_slice(&count.to_be_bytes());
            payload.extend_from_slice(&delta.to_be_bytes());
        }
        crate::mp4box::full_box(b"stts", 0, 0, &payload)
    }

    #[test]
    fn constant_framerate_needs_one_nonzero_sample_delta() {
        const TIMESCALE: u32 = 1_200_000;
        let table = |entries: &[(u32, u32)]| stts_box(entries.len() as u32, entries);
        assert_eq!(
            constant_framerate_q16(&table(&[(10, 50_000)]), TIMESCALE),
            Some(24 << 16)
        );
        assert_eq!(
            constant_framerate_q16(&table(&[(9, 50_000), (1, 40_000)]), TIMESCALE),
            None,
            "a varying delta"
        );
        assert_eq!(constant_framerate_q16(&table(&[(10, 0)]), TIMESCALE), None);
        assert_eq!(
            constant_framerate_q16(&table(&[(10, 1)]), u32::MAX),
            None,
            "a rate past the Q16 range"
        );
        assert_eq!(constant_framerate_q16(&table(&[]), TIMESCALE), None);
        assert_eq!(
            constant_framerate_q16(&stts_box(u32::MAX, &[(10, 50_000)]), TIMESCALE),
            None,
            "an entry count past the box"
        );
    }

    fn start(trak: &[u8]) -> u64 {
        edit_media_start(&edit_list(trak))
    }

    #[test]
    fn edit_media_start_reads_the_first_edit() {
        // v0: segment_duration u32, media_time i32, rate.
        let v0 = [
            48_000u32.to_be_bytes(),
            1024u32.to_be_bytes(),
            0x0001_0000u32.to_be_bytes(),
        ]
        .concat();
        assert_eq!(start(&trak_with_elst(0, 1, &v0)), 1024);
        // v1: segment_duration u64, media_time i64, rate.
        let v1 = [
            48_000u64.to_be_bytes().as_slice(),
            &2048u64.to_be_bytes(),
            &0x0001_0000u32.to_be_bytes(),
        ]
        .concat();
        assert_eq!(start(&trak_with_elst(1, 1, &v1)), 2048);
    }

    #[test]
    fn edit_media_start_ignores_delays_and_malformed_lists() {
        assert_eq!(start(&[]), 0, "no edit list");
        // An empty edit (media_time -1) delays the presentation; it trims nothing.
        let empty = [
            1000u32.to_be_bytes(),
            u32::MAX.to_be_bytes(),
            0x0001_0000u32.to_be_bytes(),
        ]
        .concat();
        assert_eq!(start(&trak_with_elst(0, 1, &empty)), 0);
        // No entries, a truncated entry, and an unknown version all fall back.
        assert_eq!(start(&trak_with_elst(0, 0, &[])), 0);
        assert_eq!(start(&trak_with_elst(0, 1, &[0, 0, 1])), 0);
        assert_eq!(start(&trak_with_elst(2, 1, &[0; 20])), 0);
    }

    #[test]
    fn edit_media_stop_ends_the_only_media_edit() {
        const MOVIE_TIMESCALE: u32 = 1000;
        const MEDIA_TIMESCALE: u32 = 12_800;
        let edit = |duration: u32, media_time: u32| {
            [
                duration.to_be_bytes(),
                media_time.to_be_bytes(),
                0x0001_0000u32.to_be_bytes(),
            ]
            .concat()
        };
        let stop = |trak: &[u8]| {
            presented_spans(&edit_list(trak), MOVIE_TIMESCALE, MEDIA_TIMESCALE)
                .and_then(|spans| spans.last().map(|span| span.end))
        };
        // 710 ms is 9088 media ticks past the edit's start
        assert_eq!(stop(&trak_with_elst(0, 1, &edit(710, 3584))), Some(12_672));
        assert_eq!(stop(&[]), None, "no edit list");
        assert_eq!(stop(&trak_with_elst(0, 1, &edit(0, 3584))), None, "open");
        assert_eq!(
            stop(&trak_with_elst(0, 1, &edit(710, u32::MAX))),
            None,
            "an empty edit"
        );
        let two_edits = [edit(710, 3584), edit(710, 0)].concat();
        assert_eq!(stop(&trak_with_elst(0, 2, &two_edits)), None);
        assert_eq!(
            presented_spans(
                &edit_list(&trak_with_elst(0, 1, &edit(710, 3584))),
                0,
                MEDIA_TIMESCALE
            ),
            None,
            "a zero movie timescale"
        );
    }

    #[test]
    fn two_edits_present_their_media_back_to_back() {
        const MOVIE_TIMESCALE: u32 = 1000;
        const MEDIA_TIMESCALE: u32 = 12_288;
        const FIRST_EDIT_MEDIA: u32 = 1024;
        const FIRST_EDIT_MS: u32 = 750;
        const SECOND_EDIT_MEDIA: u32 = 13_312;
        const SECOND_EDIT_MS: u32 = 1000;
        const EMPTY_EDIT_MS: u32 = 500;
        // every duration here is a whole number of media ticks
        let ticks_in =
            |ms: u32| u64::from(ms) * u64::from(MEDIA_TIMESCALE) / u64::from(MOVIE_TIMESCALE);
        let ns = |ticks: u64| ticks_to_ns(ticks, MEDIA_TIMESCALE);
        let edit = |duration: u32, media_time: u32| {
            [
                duration.to_be_bytes(),
                media_time.to_be_bytes(),
                0x0001_0000u32.to_be_bytes(),
            ]
            .concat()
        };
        let elst = [
            edit(EMPTY_EDIT_MS, u32::MAX),
            edit(FIRST_EDIT_MS, FIRST_EDIT_MEDIA),
            edit(SECOND_EDIT_MS, SECOND_EDIT_MEDIA),
        ]
        .concat();
        let edits = edit_list(&trak_with_elst(0, 3, &elst));
        let spans =
            presented_spans(&edits, MOVIE_TIMESCALE, MEDIA_TIMESCALE).expect("two media edits");
        let first_end = u64::from(FIRST_EDIT_MEDIA) + ticks_in(FIRST_EDIT_MS);
        let second_end = u64::from(SECOND_EDIT_MEDIA) + ticks_in(SECOND_EDIT_MS);
        assert_eq!(
            spans,
            [
                u64::from(FIRST_EDIT_MEDIA)..first_end,
                u64::from(SECOND_EDIT_MEDIA)..second_end
            ]
        );
        let track = TrackHeader {
            track_id: 1,
            timescale: MEDIA_TIMESCALE,
            kind: TrackKind::Video {
                codec: VideoCodec::H264,
                width: 64,
                height: 48,
                param_sets: Vec::new(),
            },
            cenc: None,
            tags: TagList::default(),
            media_start: edit_media_start(&edits),
            media_stop: Some(second_end),
            edit_gaps: edit_gaps(&spans, MEDIA_TIMESCALE),
            replayed_edits: Vec::new(),
            trex_defaults: SampleDefaults::default(),
            clean_aperture: None,
        };
        let gap_ns = ns(u64::from(SECOND_EDIT_MEDIA)) - ns(first_end);
        let segment = track.presentation_segment();
        assert_eq!(segment.start, ns(u64::from(FIRST_EDIT_MEDIA)));
        let stop = ns(second_end) - gap_ns;
        assert_eq!(segment.stop, Some(stop));
        let sample = |pts: u64, dts: u64| Sample {
            annexb: Vec::new(),
            pts_ns: ns(pts),
            dts_ns: ns(dts),
            duration_ns: 0,
            keyframe: false,
        };
        let in_first = track.edited_sample(sample(first_end - 1, first_end - 2));
        assert_eq!(
            (in_first.pts_ns, in_first.dts_ns),
            (ns(first_end - 1), ns(first_end - 2))
        );
        let in_second = track.edited_sample(sample(u64::from(SECOND_EDIT_MEDIA), first_end + 1));
        assert_eq!(
            in_second.pts_ns,
            ns(first_end),
            "the second edit follows the first"
        );
        assert_eq!(
            in_second.dts_ns,
            ns(first_end),
            "a decode time in the gap closes up"
        );
        let in_gap = track.edited_sample(sample(first_end, first_end));
        assert!(!segment.presents(in_gap.pts_ns), "the gap is cut");
        assert_eq!(in_gap.dts_ns, ns(first_end));
    }
    #[test]
    fn a_replayed_edit_list_presents_its_media_again() {
        // Airplane.mov's timescale and frame spacing
        const TIMESCALE: u32 = 600;
        const FRAME_TICKS: u32 = 40;
        let edit = |duration: u32, media_time: u32| {
            [
                duration.to_be_bytes(),
                media_time.to_be_bytes(),
                0x0001_0000u32.to_be_bytes(),
            ]
            .concat()
        };
        // both frames, the second again, then the third
        let elst = [
            edit(2 * FRAME_TICKS, 0),
            edit(FRAME_TICKS, FRAME_TICKS),
            edit(FRAME_TICKS, 2 * FRAME_TICKS),
        ]
        .concat();
        let edits = edit_list(&trak_with_elst(0, 3, &elst));
        assert_eq!(presented_spans(&edits, TIMESCALE, TIMESCALE), None);
        let replayed = replayed_edits(&edits, TIMESCALE, TIMESCALE).expect("three media edits");
        let track = TrackHeader {
            track_id: 1,
            timescale: TIMESCALE,
            kind: TrackKind::Video {
                codec: VideoCodec::H264,
                width: 64,
                height: 48,
                param_sets: Vec::new(),
            },
            cenc: None,
            tags: TagList::default(),
            media_start: edit_media_start(&edits),
            media_stop: replayed_media_stop(edit_media_start(&edits), &replayed),
            edit_gaps: Vec::new(),
            replayed_edits: replayed,
            trex_defaults: SampleDefaults::default(),
            clean_aperture: None,
        };
        let ns = |ticks: u32| ticks_to_ns(u64::from(ticks), TIMESCALE);
        let sample = |index: u32, keyframe: bool| Sample {
            annexb: Vec::from([index as u8]),
            pts_ns: ns(index * FRAME_TICKS),
            dts_ns: ns(index * FRAME_TICKS),
            duration_ns: ns(FRAME_TICKS),
            keyframe,
        };
        let samples = Vec::from([sample(0, true), sample(1, false), sample(2, true)]);
        let segment = track.presentation_segment();
        assert_eq!(segment.stop, Some(ns(4 * FRAME_TICKS)));

        let edited = track.edited_samples(samples);
        let order: Vec<u8> = edited.iter().map(|sample| sample.annexb[0]).collect();
        assert_eq!(
            order,
            [0, 1, 0, 1, 2],
            "the second edit decodes from the keyframe"
        );
        let presented: Vec<u64> = edited
            .iter()
            .map(|sample| sample.pts_ns)
            .filter(|pts| segment.presents(*pts))
            .collect();
        assert_eq!(
            presented,
            [0, 1, 2, 3].map(|frame| ns(frame * FRAME_TICKS)),
            "the reference decoded again for the replay is not shown"
        );
    }

    const CLAP_PICTURE: (u16, u16) = (64, 48);

    fn clap_payload(fields: [i32; 8]) -> Vec<u8> {
        fields
            .iter()
            .flat_map(|field| field.to_be_bytes())
            .collect()
    }

    // one avc1 entry of the picture size, holding only the given clap payload
    fn entries_with_clap(clap: &[u8]) -> Vec<u8> {
        let (width, height) = CLAP_PICTURE;
        let mut entry = alloc::vec![0u8; VISUAL_SAMPLE_ENTRY_CHILDREN_AT];
        entry[VISUAL_SAMPLE_ENTRY_WIDTH_AT..VISUAL_SAMPLE_ENTRY_WIDTH_AT + 2]
            .copy_from_slice(&width.to_be_bytes());
        entry[VISUAL_SAMPLE_ENTRY_HEIGHT_AT..VISUAL_SAMPLE_ENTRY_HEIGHT_AT + 2]
            .copy_from_slice(&height.to_be_bytes());
        entry.extend_from_slice(&crate::mp4box::mp4_box(b"clap", clap));
        crate::mp4box::mp4_box(b"avc1", &entry)
    }

    #[test]
    fn reads_the_clean_aperture_as_ffmpeg_crops_it() {
        // 45 columns centred on (64 - 1) / 2 + 3 = 34.5 span 12.5 to 56.5, truncated to 12 and 56,
        // so 12 columns go on the left and 63 - 56 = 7 on the right.
        // 33 rows centred on (48 - 1) / 2 - 2 = 21.5 span 5.5 to 37.5, so 5 rows go on top and 10 below.
        let entries = entries_with_clap(&clap_payload([45, 1, 33, 1, 3, 1, -2, 1]));
        assert_eq!(
            clean_aperture(&entries),
            Some(CropRectangle {
                left: 12,
                top: 5,
                width: 45,
                height: 33
            })
        );
        // the same aperture in halves: 90/2 columns and 66/2 rows, offsets 6/2 and -4/2
        let halves = entries_with_clap(&clap_payload([90, 2, 66, 2, 6, 2, -4, 2]));
        assert_eq!(clean_aperture(&halves), clean_aperture(&entries));
    }

    #[test]
    fn a_clean_aperture_of_the_whole_picture_cuts_nothing() {
        let (width, height) = CLAP_PICTURE;
        let whole = [i32::from(width), 1, i32::from(height), 1, 0, 1, 0, 1];
        assert_eq!(
            clean_aperture(&entries_with_clap(&clap_payload(whole))),
            None
        );
    }

    #[test]
    fn refuses_a_clean_aperture_that_does_not_fit() {
        let (width, height) = (u32::from(CLAP_PICTURE.0), u32::from(CLAP_PICTURE.1));
        let wider_than_the_picture = i32::from(CLAP_PICTURE.0) + 2;
        for (fields, case) in [
            ([45, 0, 33, 1, 0, 1, 0, 1], "a zero aperture denominator"),
            ([45, 1, 33, 1, 0, 0, 0, 1], "a zero offset denominator"),
            ([45, 1, 33, 1, 0, -1, 0, 1], "a negative offset denominator"),
            ([-45, 1, 33, 1, 0, 1, 0, 1], "a negative aperture"),
            (
                [wider_than_the_picture, 1, 33, 1, 0, 1, 0, 1],
                "an aperture wider than the picture",
            ),
            ([45, 1, 33, 1, 20, 1, 0, 1], "an offset past the right edge"),
            ([45, 1, 33, 1, 0, 1, -20, 1], "an offset past the top edge"),
            ([0, 1, 33, 1, 0, 1, 0, 1], "an empty aperture"),
        ] {
            let clap = clap_payload(fields);
            assert!(
                clap_rectangle(&clap, width, height).is_err(),
                "{case} is refused"
            );
            assert_eq!(
                clean_aperture(&entries_with_clap(&clap)),
                None,
                "{case} crops nothing"
            );
        }
        let truncated = clap_payload([45, 1, 33, 1, 3, 1, -2, 1]);
        let truncated = &truncated[..truncated.len() - 1];
        assert!(clap_rectangle(truncated, width, height).is_err());
        assert_eq!(clean_aperture(&entries_with_clap(truncated)), None);
    }

    use alloc::vec;

    /// The chunk splitter cuts a CMAF segment after every `mdat` whatever the
    /// arrival boundaries (fed one byte at a time here, the worst case), hands
    /// every fed byte back exactly once and in order, and leaves a trailing box
    /// for `finish`.
    #[test]
    fn cmaf_chunker_splits_after_every_mdat() {
        use crate::mp4box::mp4_box;
        let styp = mp4_box(b"styp", b"cmfscmff");
        let chunk = |n: u8| [mp4_box(b"moof", &[n; 12]), mp4_box(b"mdat", &[n; 20])].concat();
        let mfra = mp4_box(b"mfra", &[0u8; 8]);
        let segment = [styp.clone(), chunk(1), chunk(2), chunk(3), mfra.clone()].concat();

        let mut chunker = CmafChunker::new(1024);
        let mut got = Vec::new();
        for byte in &segment {
            chunker.feed(&[*byte]).expect("under the cap");
            while let Some(c) = chunker.next_chunk().expect("well-formed boxes") {
                got.push(c);
            }
        }
        assert_eq!(
            got,
            vec![[styp, chunk(1)].concat(), chunk(2), chunk(3)],
            "the leading styp rides the first chunk, then one chunk per moof+mdat"
        );
        assert_eq!(
            chunker.finish(),
            Some(mfra),
            "the trailing box comes out last"
        );
        assert_eq!(chunker.finish(), None, "and only once");
        assert_eq!(
            got.concat().len() + 16,
            segment.len(),
            "no byte was dropped"
        );
    }

    /// The cap is the splitter's DoS bound: a box declaring more than it fails
    /// before the bytes are waited on, and so does a pending run growing past it.
    #[test]
    fn cmaf_chunker_bounds_declared_and_pending_sizes() {
        let mut chunker = CmafChunker::new(64);
        // An mdat claiming 1 MiB: rejected on the header, not buffered toward.
        chunker
            .feed(&[0x00, 0x10, 0x00, 0x00, b'm', b'd', b'a', b't'])
            .expect("8 bytes fit");
        assert!(
            chunker.next_chunk().is_err(),
            "over-cap box size fails loud"
        );

        let mut chunker = CmafChunker::new(64);
        assert!(
            chunker.feed(&[0u8; 65]).is_err(),
            "over-cap pending run fails"
        );
    }

    #[test]
    fn avcc_to_annexb_round_trips_length_prefixes() {
        let avcc = [0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3];
        let annexb = avcc_to_annexb(&avcc).unwrap();
        assert_eq!(annexb, vec![0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 3]);
        // a truncated NALU fails loud
        assert!(avcc_to_annexb(&[0, 0, 0, 9, 1]).is_err());
    }

    #[test]
    fn trun_parser_reads_the_writer_profile() {
        // flags 0x701: data offset + duration + size + flags, one sample.
        let mut p = vec![0u8, 0, 7, 1];
        p.extend_from_slice(&1u32.to_be_bytes()); // count
        p.extend_from_slice(&120u32.to_be_bytes()); // data offset
        p.extend_from_slice(&3000u32.to_be_bytes()); // duration
        p.extend_from_slice(&77u32.to_be_bytes()); // size
        p.extend_from_slice(&0x0200_0000u32.to_be_bytes()); // sample flags
        let samples = parse_trun(&p, SampleDefaults::default(), 0).unwrap();
        assert_eq!(
            samples,
            vec![TrunSample {
                size: 77,
                duration: 3000,
                composition_offset: 0,
                sync: Some(true),
            }]
        );
    }

    #[test]
    fn sps_detection_reads_the_first_nal_type() {
        assert!(starts_with_param_set(
            &[0, 0, 0, 1, 0x67, 0xAA],
            VideoCodec::H264
        ));
        assert!(!starts_with_param_set(
            &[0, 0, 0, 1, 0x65, 0xAA],
            VideoCodec::H264
        ));
        assert!(starts_with_param_set(
            &[0, 0, 0, 1, 0x40, 0x01],
            VideoCodec::H265
        ));
        assert!(!starts_with_param_set(
            &[0, 0, 0, 1, 0x26, 0x01],
            VideoCodec::H265
        ));
    }

    #[test]
    fn hvcc_parser_recovers_arrays_in_order() {
        let vps: &[u8] = &[0x40, 0x01, 0xAA];
        let sps: &[u8] = &[0x42, 0x01, 0xBB, 0xCC];
        let pps: &[u8] = &[0x44, 0x01, 0xDD];
        let mut p = vec![0u8; 22];
        p[0] = 1; // configuration version
        p.push(3); // numOfArrays at offset 22
        for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
            p.push(0x80 | ty);
            p.extend_from_slice(&1u16.to_be_bytes());
            p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
            p.extend_from_slice(nalu);
        }
        let sets = parse_hvcc(&p).unwrap().parameter_sets;
        assert_eq!(sets, vec![vps.to_vec(), sps.to_vec(), pps.to_vec()]);
    }

    // ffmpeg testsrc 64x48, libx264 -bf 2 -g 6, -movflags frag_keyframe+empty_moov
    const BFRAME_FRAGMENTED: &[u8] =
        include_bytes!("../tests/fixtures/h264_64x48_bframes_frag.mp4");
    // ffprobe -show_entries packet=pts,dts -of csv=p=0, in track timescale ticks
    const BFRAME_FRAGMENTED_PTS_DTS: &str =
        include_str!("../tests/fixtures/h264_64x48_bframes_frag.mp4.pts_dts");

    fn probed_pts_dts_ns(timescale: u32) -> Vec<(u64, u64)> {
        let ticks_ns = |ticks: &str| ticks_to_ns(ticks.parse().unwrap(), timescale);
        BFRAME_FRAGMENTED_PTS_DTS
            .lines()
            .map(|line| {
                let (pts, dts) = line.split_once(',').unwrap();
                (ticks_ns(pts), ticks_ns(dts))
            })
            .collect()
    }

    #[test]
    fn parse_fragments_applies_composition_offsets() {
        let header = parse_header(BFRAME_FRAGMENTED).unwrap();
        let samples = parse_fragments(
            BFRAME_FRAGMENTED,
            header.timescale,
            header.codec,
            None,
            0,
            &mut FragmentStarts::default(),
            None,
        )
        .unwrap();
        let pts_dts: Vec<(u64, u64)> = samples.iter().map(|s| (s.pts_ns, s.dts_ns)).collect();
        assert_eq!(pts_dts, probed_pts_dts_ns(header.timescale));
    }

    #[test]
    fn parse_fragments_multi_applies_composition_offsets() {
        let tracks = parse_all_tracks(BFRAME_FRAGMENTED).unwrap();
        let samples = parse_fragments_multi(
            BFRAME_FRAGMENTED,
            &tracks,
            0,
            &mut FragmentStarts::default(),
            None,
        )
        .unwrap();
        let pts_dts: Vec<(u64, u64)> = samples.iter().map(|(_, s)| (s.pts_ns, s.dts_ns)).collect();
        assert_eq!(pts_dts, probed_pts_dts_ns(tracks[0].timescale));
    }

    #[test]
    fn parse_trun_reads_composition_offset_signed_in_both_versions() {
        // flags 0xA01: data offset + sample size + composition offset.
        const OFFSET_FIELD: u32 = 0xFFFF_FE00;
        let build = |version: u8| {
            let mut t = alloc::vec![version, 0x00, 0x0A, 0x01];
            t.extend_from_slice(&1u32.to_be_bytes()); // sample count
            t.extend_from_slice(&0u32.to_be_bytes()); // data offset
            t.extend_from_slice(&1000u32.to_be_bytes()); // sample size
            t.extend_from_slice(&OFFSET_FIELD.to_be_bytes());
            t
        };
        let v0 = parse_trun(&build(0), SampleDefaults::default(), 0).expect("v0 parses");
        let v1 = parse_trun(&build(1), SampleDefaults::default(), 0).expect("v1 parses");
        let sample = |composition_offset| TrunSample {
            size: 1000,
            duration: 0,
            composition_offset,
            sync: None,
        };
        let expected = alloc::vec![sample(i64::from(OFFSET_FIELD as i32))];
        assert_eq!(v0, expected);
        assert_eq!(v1, expected);
    }

    #[test]
    fn parse_trun_rejects_oversized_count() {
        // flags 0x201 (data-offset + sizes), a huge count but only one sample's
        // worth of bytes: reject instead of reserving gigabytes.
        let mut t = alloc::vec![0u8, 0x00, 0x02, 0x01];
        t.extend_from_slice(&u32::MAX.to_be_bytes()); // count
        t.extend_from_slice(&0u32.to_be_bytes()); // data offset
        t.extend_from_slice(&16u32.to_be_bytes()); // a single sample size
        assert!(parse_trun(&t, SampleDefaults::default(), 0).is_err());
    }

    #[test]
    fn parse_trun_takes_first_sample_flags_then_the_defaults() {
        // flags 0x205: data offset + first sample flags + sizes, three samples.
        const DEPENDS_ON_NONE: u32 = 0x0200_0000;
        let mut t = alloc::vec![0u8, 0x00, 0x02, 0x05];
        t.extend_from_slice(&3u32.to_be_bytes()); // count
        t.extend_from_slice(&0u32.to_be_bytes()); // data offset
        t.extend_from_slice(&DEPENDS_ON_NONE.to_be_bytes()); // first sample flags
        for _ in 0..3 {
            t.extend_from_slice(&16u32.to_be_bytes());
        }
        let defaults = SampleDefaults {
            flags: Some(NON_SYNC),
            ..SampleDefaults::default()
        };
        let sync: Vec<_> = parse_trun(&t, defaults, 0)
            .unwrap()
            .iter()
            .map(|s| s.sync)
            .collect();
        assert_eq!(sync, vec![Some(true), Some(false), Some(false)]);
    }

    #[test]
    fn sample_that_depends_on_others_is_not_sync() {
        assert!(sample_flags_sync(0x0200_0000));
        assert!(!sample_flags_sync(DEPENDS_ON_OTHERS));
        assert!(!sample_flags_sync(NON_SYNC));
    }

    #[test]
    fn fragment_keyframe_follows_the_flags_except_for_intra_only_codecs() {
        let idr = [0, 0, 0, 1, 0x65, 0x88];
        assert!(!fragment_video_keyframe(
            VideoCodec::H264,
            Some(false),
            &idr
        ));
        assert!(fragment_video_keyframe(VideoCodec::H264, None, &idr));
        assert!(fragment_video_keyframe(
            VideoCodec::Mjpeg,
            Some(false),
            &[0xFF, 0xD8]
        ));
        assert!(fragment_video_keyframe(VideoCodec::DnxHd, None, &[0; 4]));
    }

    /// A minimal progressive (`moov` + `mdat`, no `moof`) file with two AVCC
    /// samples in one chunk parses to two Annex-B samples with the right sizes,
    /// timing, and sync flag (sample 1 only, from `stss`).
    #[test]
    fn parse_progressive_reads_stbl_sample_tables() {
        use crate::mp4box::{full_box, mp4_box};
        // Two AVCC samples: [len=2][0x65 IDR][0xAA], [len=2][0x41 non-IDR][0xBB].
        let mut mdat_body = Vec::new();
        for nal in [[0x65u8, 0xAA], [0x41, 0xBB]] {
            mdat_body.extend_from_slice(&2u32.to_be_bytes());
            mdat_body.extend_from_slice(&nal);
        }
        let sample_size = 6u32; // 4-byte length prefix + 2-byte NAL

        let stsz = {
            let mut b = alloc::vec![0u8; 8]; // default_size = 0, then count
            b[4..8].copy_from_slice(&2u32.to_be_bytes());
            b.extend_from_slice(&sample_size.to_be_bytes());
            b.extend_from_slice(&sample_size.to_be_bytes());
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one run
            b.extend_from_slice(&2u32.to_be_bytes()); // count
            b.extend_from_slice(&1000u32.to_be_bytes()); // delta
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one entry
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
            b.extend_from_slice(&2u32.to_be_bytes()); // samples_per_chunk = 2
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_desc_index
            full_box(b"stsc", 0, 0, &b)
        };
        let stss = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one sync sample
            b.extend_from_slice(&1u32.to_be_bytes()); // sample number 1 (1-based)
            full_box(b"stss", 0, 0, &b)
        };
        // stco offset is filled once the moov length is known (it is constant in
        // the offset value, so a placeholder build gives the right length).
        let build = |chunk_off: u32| {
            let mut stco_body = 1u32.to_be_bytes().to_vec();
            stco_body.extend_from_slice(&chunk_off.to_be_bytes());
            let stco = full_box(b"stco", 0, 0, &stco_body);
            let mut stbl = Vec::new();
            for t in [&stsz, &stts, &stsc, &stco, &stss] {
                stbl.extend_from_slice(t);
            }
            let stbl = mp4_box(b"stbl", &stbl);
            let minf = mp4_box(b"minf", &stbl);
            let mdia = mp4_box(b"mdia", &minf);
            let trak = mp4_box(b"trak", &mdia);
            mp4_box(b"moov", &trak)
        };
        let moov_len = build(0).len();
        let chunk_off = (moov_len + 8) as u32; // mdat payload starts after its header
        let mut file = build(chunk_off);
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_body));

        let samples = parse_progressive(&file, 1000).expect("progressive parse");
        assert_eq!(samples.len(), 2);
        // AVCC length prefixes became Annex-B start codes.
        assert_eq!(samples[0].annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert_eq!(samples[1].annexb, alloc::vec![0, 0, 0, 1, 0x41, 0xBB]);
        assert!(samples[0].keyframe, "sample 1 is in stss");
        assert!(!samples[1].keyframe, "sample 2 is not in stss");
        assert_eq!(samples[0].pts_ns, 0);
        assert_eq!(samples[1].pts_ns, 1_000_000_000); // 1000 / timescale 1000 s

        // a file cut inside the last sample keeps the whole ones
        let truncated = parse_progressive(&file[..file.len() - 1], 1000).expect("truncated parse");
        assert_eq!(truncated.len(), 1);
        assert_eq!(truncated[0].annexb, samples[0].annexb);

        // a long-tail video track also keeps what remains of the cut sample
        let cut_file = &file[..file.len() - 1];
        let stbl = find_path(cut_file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]).unwrap();
        let tagged =
            parse_progressive_track(cut_file, stbl, 1000, SampleFraming::TaggedVideo).unwrap();
        let second_sample_at = sample_size as usize;
        assert_eq!(tagged.len(), 2);
        assert_eq!(
            tagged[1].annexb,
            mdat_body[second_sample_at..mdat_body.len() - 1]
        );
    }

    #[test]
    fn composition_offsets_read_signed_in_version_0() {
        use crate::mp4box::full_box;
        const NEGATIVE_OFFSET: i32 = -3003;
        let mut runs = 2u32.to_be_bytes().to_vec();
        for (count, offset) in [(1u32, 3003i32), (2, NEGATIVE_OFFSET)] {
            runs.extend_from_slice(&count.to_be_bytes());
            runs.extend_from_slice(&offset.to_be_bytes());
        }
        let stbl = full_box(b"ctts", 0, 0, &runs);
        assert_eq!(
            composition_offsets(&stbl, 4).expect("ctts parses"),
            alloc::vec![
                3003,
                i64::from(NEGATIVE_OFFSET),
                i64::from(NEGATIVE_OFFSET),
                0
            ]
        );
    }

    #[test]
    fn parse_progressive_walks_a_large_stsc() {
        use crate::mp4box::{full_box, mp4_box};
        const CHUNKS: u32 = 20_000;
        const SAMPLE_LEN: u32 = 5;
        // One stsc entry per chunk, alternating one and two samples per chunk.
        let samples_in = |chunk: u32| 1 + chunk % 2;
        let sample_count: u32 = (0..CHUNKS).map(samples_in).sum();
        let mut mdat_body = Vec::new();
        for sample in 0..sample_count {
            mdat_body.extend_from_slice(&1u32.to_be_bytes());
            mdat_body.push(sample as u8);
        }
        let stsz = {
            let mut b = 0u32.to_be_bytes().to_vec();
            b.extend_from_slice(&sample_count.to_be_bytes());
            for _ in 0..sample_count {
                b.extend_from_slice(&SAMPLE_LEN.to_be_bytes());
            }
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&sample_count.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = {
            let mut b = CHUNKS.to_be_bytes().to_vec();
            for chunk in 0..CHUNKS {
                b.extend_from_slice(&(chunk + 1).to_be_bytes());
                b.extend_from_slice(&samples_in(chunk).to_be_bytes());
                b.extend_from_slice(&1u32.to_be_bytes());
            }
            full_box(b"stsc", 0, 0, &b)
        };
        let build = |mdat_start: u32| {
            let mut stco_body = CHUNKS.to_be_bytes().to_vec();
            let mut at = mdat_start;
            for chunk in 0..CHUNKS {
                stco_body.extend_from_slice(&at.to_be_bytes());
                at += samples_in(chunk) * SAMPLE_LEN;
            }
            let stco = full_box(b"stco", 0, 0, &stco_body);
            let stbl = mp4_box(b"stbl", &[&stsz[..], &stts, &stsc, &stco].concat());
            let trak = mp4_box(b"trak", &mp4_box(b"mdia", &mp4_box(b"minf", &stbl)));
            mp4_box(b"moov", &trak)
        };
        let mdat_start = (build(0).len() + 8) as u32;
        let mut file = build(mdat_start);
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_body));

        let samples = parse_progressive(&file, 1).expect("progressive parse");
        assert_eq!(samples.len(), sample_count as usize);
        for (index, sample) in samples.iter().enumerate() {
            assert_eq!(sample.annexb, alloc::vec![0, 0, 0, 1, index as u8]);
        }
    }

    /// An `mp4v` visual sample entry whose esds objectTypeIndication is 0x20
    /// (Visual ISO/IEC 14496-2) parses to `VideoCodec::Mpeg4Part2` with the VOL
    /// header (the DecoderSpecificInfo) as its single parameter set; a different
    /// objectTypeIndication in an mp4v box is rejected so another codec is not
    /// mistagged as MPEG-4 Part 2.
    #[test]
    fn parse_header_reads_mp4v_mpeg4_part2() {
        use crate::mp4box::{full_box, mp4_box};

        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        // esds with a chosen objectTypeIndication (first DCD byte) and DSI.
        let esds = |oti: u8, dsi_bytes: &[u8]| {
            let dsi = descriptor(0x05, dsi_bytes);
            let mut dcd_body = vec![0u8; 13];
            dcd_body[0] = oti;
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            full_box(b"esds", 0, 0, &descriptor(0x03, &es_body))
        };
        let tkhd = {
            let mut c = vec![0u8; 80];
            c[72..76].copy_from_slice(&(320u32 << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(240u32 << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = {
            let mut c = vec![0u8; 16];
            c[8..12].copy_from_slice(&1000u32.to_be_bytes()); // timescale
            c[12..16].copy_from_slice(&1000u32.to_be_bytes()); // duration
            full_box(b"mdhd", 0, 0, &c)
        };
        // The VOL header carries its own 3-byte start code (VOS start code 0xB0).
        let vol: &[u8] = &[0x00, 0x00, 0x01, 0xB0, 0x08];
        let build = |oti: u8| {
            let mp4v = {
                let mut p = vec![0u8; 78];
                p.extend_from_slice(&esds(oti, vol));
                mp4_box(b"mp4v", &p)
            };
            let mut stsd_body = 1u32.to_be_bytes().to_vec(); // entry count
            stsd_body.extend_from_slice(&mp4v);
            let stsd = full_box(b"stsd", 0, 0, &stsd_body);
            let stbl = mp4_box(b"stbl", &stsd);
            let minf = mp4_box(b"minf", &stbl);
            let mdia = mp4_box(b"mdia", &[mdhd.clone(), minf].concat());
            let trak = mp4_box(b"trak", &[tkhd.clone(), mdia].concat());
            mp4_box(b"moov", &trak)
        };

        let header = parse_header(&build(0x20)).expect("mp4v/esds OTI 0x20 parses");
        assert_eq!(header.codec, VideoCodec::Mpeg4Part2);
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
        assert_eq!(
            header.param_sets,
            vec![vol.to_vec()],
            "VOL header is the config"
        );

        assert!(
            parse_header(&build(0x21)).is_err(),
            "non-Visual objectTypeIndication rejected"
        );
        // the MPEG-2 Visual profiles of ISO/IEC 14496-1, then MPEG-1 Visual
        for object_type in [0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x6A] {
            let header = parse_header(&build(object_type))
                .unwrap_or_else(|e| panic!("objectTypeIndication {object_type:#x}: {e:?}"));
            assert_eq!(header.codec, VideoCodec::Mpeg2, "{object_type:#x}");
        }
    }

    /// MPEG-4 Part 2 config framing differs from H.264/H.265: the VOL header
    /// already carries 3-byte start codes, so it is prepended verbatim (no
    /// injected 4-byte Annex-B start code), and an access unit already opening
    /// with a VOS/VOL start code counts as carrying its parameter set.
    #[test]
    fn mpeg4_part2_config_framing() {
        let vol: Vec<u8> = vec![0x00, 0x00, 0x01, 0xB0, 0x08];
        let frame: Vec<u8> = vec![0x00, 0x00, 0x01, 0xB6, 0x00]; // I-VOP

        let out = prepend_param_sets(&frame, core::slice::from_ref(&vol), VideoCodec::Mpeg4Part2);
        assert_eq!(
            out,
            [vol.clone(), frame.clone()].concat(),
            "VOL prepended verbatim"
        );

        // H.264 param sets still get a 4-byte start code injected.
        let sps: Vec<u8> = vec![0x67, 0x42];
        let h264 = prepend_param_sets(&frame, core::slice::from_ref(&sps), VideoCodec::H264);
        assert_eq!(
            h264[..4],
            [0, 0, 0, 1],
            "H.264 set gets an Annex-B start code"
        );

        assert!(
            starts_with_param_set(&vol, VideoCodec::Mpeg4Part2),
            "VOS start code detected"
        );
        assert!(
            !starts_with_param_set(&frame, VideoCodec::Mpeg4Part2),
            "an I-VOP is not config"
        );
    }

    /// A two-track fragmented file (an H.264 `vide` trak + an AAC `soun` trak,
    /// then one `moof`+`mdat` per track) parses to two [`TrackHeader`]s with the
    /// right codec/geometry/timescale, and [`parse_fragments_multi`] routes each
    /// fragment to its `track_ID`, de-framing video to Annex-B and passing audio
    /// through. Builds the boxes directly so the test stays a lib unit test.
    #[test]
    fn parse_all_tracks_and_fragments_route_by_track_id() {
        use crate::mp4box::{ftyp, full_box, mp4_box};

        // --- box builders the parser's offsets expect ---------------------
        // tkhd v0: track_ID at payload offset 12, width/height 16.16 at 76/80.
        let tkhd = |track_id: u32, w: u32, h: u32| {
            let mut c = alloc::vec![0u8; 80]; // content after the version/flags
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        // mdhd v0: timescale at payload offset 12, duration at 16.
        let mdhd = |timescale: u32, duration: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&timescale.to_be_bytes());
            c[12..16].copy_from_slice(&duration.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        // hdlr: handler_type at payload offset 8.
        let hdlr = |handler: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(handler);
            full_box(b"hdlr", 0, 0, &c)
        };
        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = alloc::vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        let esds = |asc: &[u8]| {
            let dsi = descriptor(0x05, asc);
            let mut dcd_body = alloc::vec![0u8; 13];
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = alloc::vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            let es = descriptor(0x03, &es_body);
            full_box(b"esds", 0, 0, &es)
        };
        let avcc = |sps: &[u8], pps: &[u8]| {
            let mut p = alloc::vec![0u8; 5]; // fixed config bytes
            p.push(0xE1); // reserved bits + sps_count = 1
            p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            p.extend_from_slice(sps);
            p.push(1); // pps_count
            p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            p.extend_from_slice(pps);
            mp4_box(b"avcC", &p)
        };
        let stsd = |entry: &[u8]| {
            let mut p = 1u32.to_be_bytes().to_vec(); // entry count
            p.extend_from_slice(entry);
            full_box(b"stsd", 0, 0, &p)
        };
        let trak = |tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], stsd: &[u8]| {
            let minf = mp4_box(b"minf", &mp4_box(b"stbl", stsd));
            let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
            mp4_box(b"trak", &[tkhd, &mdia].concat())
        };

        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
        let pps: &[u8] = &[0x68, 0xce];
        let asc: &[u8] = &[0x12, 0x10];

        // avc1 sample entry: 78 fixed bytes then the avcC.
        let avc1 = {
            let mut p = alloc::vec![0u8; 78];
            p.extend_from_slice(&avcc(sps, pps));
            mp4_box(b"avc1", &p)
        };
        // mp4a sample entry: channelcount at offset 16, then 28 bytes before esds.
        let mp4a = {
            let mut p = alloc::vec![0u8; 28];
            p[16..18].copy_from_slice(&2u16.to_be_bytes());
            p.extend_from_slice(&esds(asc));
            mp4_box(b"mp4a", &p)
        };

        let video_trak = trak(
            &tkhd(1, 320, 240),
            &mdhd(90_000, 90_000), // 1 s
            &hdlr(b"vide"),
            &stsd(&avc1),
        );
        let audio_trak = trak(
            &tkhd(2, 0, 0),
            &mdhd(48_000, 48_000), // 1 s
            &hdlr(b"soun"),
            &stsd(&mp4a),
        );
        let moov = mp4_box(b"moov", &[video_trak, audio_trak].concat());

        // --- fragments: one per track, keyed by track_ID via tfhd ---------
        let tfhd = |track_id: u32| full_box(b"tfhd", 0, 0, &track_id.to_be_bytes());
        let tfdt = |base: u64| full_box(b"tfdt", 1, 0, &base.to_be_bytes());
        let trun = |dur: u32, size: u32| {
            let mut p = 1u32.to_be_bytes().to_vec(); // sample count
            p.extend_from_slice(&0u32.to_be_bytes()); // data offset
            p.extend_from_slice(&dur.to_be_bytes());
            p.extend_from_slice(&size.to_be_bytes());
            full_box(b"trun", 0, 0x000301, &p) // data-offset | duration | size
        };
        let moof = |track_id: u32, dur: u32, size: u32| {
            let traf = mp4_box(
                b"traf",
                &[tfhd(track_id), tfdt(0), trun(dur, size)].concat(),
            );
            mp4_box(b"moof", &traf)
        };

        // Video sample: one AVCC NALU (4-byte length + IDR), de-frames to Annex-B.
        let video_sample = alloc::vec![0, 0, 0, 2, 0x65, 0xAA];
        let audio_sample = alloc::vec![0x01u8, 0x02, 0x03]; // raw AAC, passed through

        let mut file = ftyp();
        file.extend_from_slice(&moov);
        file.extend_from_slice(&moof(1, 3000, video_sample.len() as u32));
        file.extend_from_slice(&mp4_box(b"mdat", &video_sample));
        file.extend_from_slice(&moof(2, 1024, audio_sample.len() as u32));
        file.extend_from_slice(&mp4_box(b"mdat", &audio_sample));

        // --- assert: two tracks parsed with the right kinds ---------------
        let tracks = parse_all_tracks(&file).expect("two-track parse");
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].track_id, 1);
        match &tracks[0].kind {
            TrackKind::Video {
                codec,
                width,
                height,
                param_sets,
            } => {
                assert_eq!(*codec, VideoCodec::H264);
                assert_eq!((*width, *height), (320, 240));
                assert_eq!(param_sets, &alloc::vec![sps.to_vec(), pps.to_vec()]);
            }
            other => panic!("track 0 should be video, got {other:?}"),
        }
        assert_eq!(tracks[1].track_id, 2);
        match &tracks[1].kind {
            TrackKind::Audio {
                format,
                channels,
                sample_rate,
                config: got,
                ..
            } => {
                assert_eq!(*format, AudioFormat::Aac);
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, 48_000);
                assert_eq!(got, asc);
            }
            other => panic!("track 1 should be audio, got {other:?}"),
        }

        // --- assert: fragments route to their track and de-frame correctly -
        let samples =
            parse_fragments_multi(&file, &tracks, 0, &mut FragmentStarts::default(), None)
                .expect("fragment routing");
        assert_eq!(samples.len(), 2);
        let (vid_id, vid) = &samples[0];
        assert_eq!(*vid_id, 1);
        assert_eq!(vid.annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert!(vid.keyframe, "IDR is a keyframe");
        let (aud_id, aud) = &samples[1];
        assert_eq!(*aud_id, 2);
        assert_eq!(aud.annexb, audio_sample, "audio passes through verbatim");
        assert!(aud.keyframe, "every audio AU is a sync sample");
    }

    /// A progressive (`moov`+`mdat`, no `moof`) two-track file (an H.264 `vide`
    /// trak + an AAC `soun` trak, each with classic `stbl` sample tables sharing a
    /// single `mdat`) parses to two tracks and `parse_progressive_multi` routes
    /// each track's samples by `track_ID`, de-framing video to Annex-B and passing
    /// audio through. The progressive analog of the fragmented test above.
    #[test]
    fn parse_progressive_multi_routes_each_track_by_id() {
        use crate::mp4box::{full_box, mp4_box};

        // --- shared leaf box builders (same offsets the parser reads) ------
        let tkhd = |track_id: u32, w: u32, h: u32| {
            let mut c = alloc::vec![0u8; 80];
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = |timescale: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&timescale.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        let hdlr = |handler: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(handler);
            full_box(b"hdlr", 0, 0, &c)
        };
        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = alloc::vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        let esds = |asc: &[u8]| {
            let dsi = descriptor(0x05, asc);
            let mut dcd_body = alloc::vec![0u8; 13];
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = alloc::vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            full_box(b"esds", 0, 0, &descriptor(0x03, &es_body))
        };
        let avcc = |sps: &[u8], pps: &[u8]| {
            let mut p = alloc::vec![0u8; 5];
            p.push(0xE1);
            p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            p.extend_from_slice(sps);
            p.push(1);
            p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            p.extend_from_slice(pps);
            mp4_box(b"avcC", &p)
        };
        let stsd = |entry: &[u8]| {
            let mut p = 1u32.to_be_bytes().to_vec();
            p.extend_from_slice(entry);
            full_box(b"stsd", 0, 0, &p)
        };
        // sample-table builders (one chunk holding all of a track's samples).
        let stsz = |sizes: &[u32]| {
            let mut b = alloc::vec![0u8; 8]; // default_size 0, then count
            b[4..8].copy_from_slice(&(sizes.len() as u32).to_be_bytes());
            for s in sizes {
                b.extend_from_slice(&s.to_be_bytes());
            }
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = |count: u32, delta: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&count.to_be_bytes());
            b.extend_from_slice(&delta.to_be_bytes());
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = |spc: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
            b.extend_from_slice(&spc.to_be_bytes()); // samples_per_chunk
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_desc_index
            full_box(b"stsc", 0, 0, &b)
        };
        let stco = |offset: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&offset.to_be_bytes());
            full_box(b"stco", 0, 0, &b)
        };
        let stss = |sample_no: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&sample_no.to_be_bytes());
            full_box(b"stss", 0, 0, &b)
        };
        let trak = |tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], stbl: &[u8]| {
            let minf = mp4_box(b"minf", &mp4_box(b"stbl", stbl));
            let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
            mp4_box(b"trak", &[tkhd, &mdia].concat())
        };

        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
        let pps: &[u8] = &[0x68, 0xce];
        let asc: &[u8] = &[0x12, 0x10];

        let avc1 = {
            let mut p = alloc::vec![0u8; 78];
            p.extend_from_slice(&avcc(sps, pps));
            mp4_box(b"avc1", &p)
        };
        let mp4a = {
            let mut p = alloc::vec![0u8; 28];
            p[16..18].copy_from_slice(&2u16.to_be_bytes());
            p.extend_from_slice(&esds(asc));
            mp4_box(b"mp4a", &p)
        };

        // Two AVCC video samples + two raw AAC samples, in one mdat each track.
        let video_samples: [&[u8]; 2] = [&[0, 0, 0, 2, 0x65, 0xAA], &[0, 0, 0, 2, 0x41, 0xBB]];
        let audio_samples: [&[u8]; 2] = [&[0xC1, 0xC2, 0xC3], &[0xD1, 0xD2]];
        let v_sizes: Vec<u32> = video_samples.iter().map(|s| s.len() as u32).collect();
        let a_sizes: Vec<u32> = audio_samples.iter().map(|s| s.len() as u32).collect();
        let v_total: u32 = v_sizes.iter().sum();

        // The moov length is constant in the (u32) chunk-offset values, so a
        // placeholder build gives the offsets to fill into the real one.
        let build = |off_v: u32, off_a: u32| {
            let v_stbl = [
                stsd(&avc1),
                stsz(&v_sizes),
                stts(2, 3000),
                stsc(2),
                stco(off_v),
                stss(1),
            ]
            .concat();
            let a_stbl = [
                stsd(&mp4a),
                stsz(&a_sizes),
                stts(2, 1024),
                stsc(2),
                stco(off_a),
            ]
            .concat();
            let video_trak = trak(&tkhd(1, 320, 240), &mdhd(90_000), &hdlr(b"vide"), &v_stbl);
            let audio_trak = trak(&tkhd(2, 0, 0), &mdhd(48_000), &hdlr(b"soun"), &a_stbl);
            mp4_box(b"moov", &[video_trak, audio_trak].concat())
        };
        let moov_len = build(0, 0).len();
        let off_v = (moov_len + 8) as u32; // mdat payload starts after its header
        let off_a = off_v + v_total;
        let mut file = build(off_v, off_a);
        let mut mdat_body = Vec::new();
        for s in video_samples.iter().chain(audio_samples.iter()) {
            mdat_body.extend_from_slice(s);
        }
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_body));

        // No moof: the multi-track progressive path must split both tracks.
        assert!(find_box(&file, b"moof").is_none(), "fixture is progressive");
        let tracks = parse_all_tracks(&file).expect("two tracks");
        assert_eq!(tracks.len(), 2);
        let samples = parse_progressive_multi(&file, &tracks).expect("progressive multi parse");
        assert_eq!(samples.len(), 4, "two video + two audio samples");
        let order: Vec<u32> = samples.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, [1, 2, 2, 1], "interleaved by decode time");

        let video: Vec<_> = samples.iter().filter(|(id, _)| *id == 1).collect();
        let audio: Vec<_> = samples.iter().filter(|(id, _)| *id == 2).collect();
        assert_eq!(video.len(), 2);
        assert_eq!(audio.len(), 2);
        // Video de-framed AVCC -> Annex-B; sample 1 (in stss) is the keyframe.
        assert_eq!(video[0].1.annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert!(video[0].1.keyframe, "sample 1 is in stss");
        assert!(!video[1].1.keyframe, "sample 2 is not in stss");
        // Audio passed through verbatim; no stss means every sample is a sync point.
        assert_eq!(audio[0].1.annexb, audio_samples[0]);
        assert_eq!(audio[1].1.annexb, audio_samples[1]);
        assert!(audio[0].1.keyframe && audio[1].1.keyframe);
    }

    #[test]
    fn parse_text_entry_recognizes_tx3g_wvtt_stpp() {
        use crate::mp4box::mp4_box;
        // tx3g + wvtt -> plain UTF-8; stpp -> TTML document. The parser only needs
        // the sample-entry box to be present in the stsd entries.
        let tx3g = mp4_box(b"tx3g", &[0u8; 8]);
        assert!(matches!(
            parse_text_entry(&tx3g),
            Some(TrackKind::Text {
                format: TextFormat::Utf8,
                sample: TextSampleFormat::Tx3g
            })
        ));
        let wvtt = mp4_box(b"wvtt", &mp4_box(b"vttC", b"WEBVTT"));
        assert!(matches!(
            parse_text_entry(&wvtt),
            Some(TrackKind::Text {
                format: TextFormat::Utf8,
                sample: TextSampleFormat::Wvtt
            })
        ));
        let stpp = mp4_box(b"stpp", &[0u8; 8]);
        assert!(matches!(
            parse_text_entry(&stpp),
            Some(TrackKind::Text {
                format: TextFormat::Ttml,
                sample: TextSampleFormat::Stpp
            })
        ));
        // A caption codec is not a text codec (it goes through parse_cc_entry).
        assert!(parse_text_entry(&mp4_box(b"c608", &[0u8; 8])).is_none());
    }

    #[test]
    fn parse_cc_entry_recognizes_c608_and_c708() {
        use crate::mp4box::mp4_box;
        assert!(matches!(
            parse_cc_entry(&mp4_box(b"c608", &[0u8; 8])),
            Some(TrackKind::ClosedCaption {
                format: ClosedCaptionFormat::Cea608
            })
        ));
        assert!(matches!(
            parse_cc_entry(&mp4_box(b"c708", &[0u8; 8])),
            Some(TrackKind::ClosedCaption {
                format: ClosedCaptionFormat::Cea708
            })
        ));
        // Another codec under the same handler declines (the track is skipped).
        assert!(parse_cc_entry(&mp4_box(b"tx3g", &[0u8; 8])).is_none());
    }

    #[test]
    fn deframe_c608_tags_cdat_and_cdt2_pairs() {
        use crate::mp4box::mp4_box;
        // cdat pairs are field 1 (cc_type 0), cdt2 pairs field 2 (cc_type 1); the
        // pairs keep their order within each atom, atoms in sample order.
        let mut sample = mp4_box(b"cdat", &[0x94, 0x20, b'H', b'I']);
        sample.extend_from_slice(&mp4_box(b"cdt2", &[0x15, 0x2C]));
        assert_eq!(
            crate::cea::parse_cc_data(&deframe_c608(&sample)),
            alloc::vec![
                CcTriple {
                    cc_type: 0,
                    b0: 0x94,
                    b1: 0x20
                },
                CcTriple {
                    cc_type: 0,
                    b0: b'H',
                    b1: b'I'
                },
                CcTriple {
                    cc_type: 1,
                    b0: 0x15,
                    b1: 0x2C
                },
            ]
        );
        // An unknown atom is skipped, not fatal.
        let mut mixed = mp4_box(b"junk", &[1, 2, 3, 4]);
        mixed.extend_from_slice(&mp4_box(b"cdat", b"OK"));
        assert_eq!(deframe_c608(&mixed).len(), 3);
    }

    #[test]
    fn deframe_c608_survives_a_lying_atom_size() {
        // A size past the sample end, a size below the 8-byte header, an odd
        // payload length, and a truncated header all yield no caption data.
        let mut lying = Vec::from(0x7FFF_FFFFu32.to_be_bytes());
        lying.extend_from_slice(b"cdat");
        lying.extend_from_slice(&[0x94, 0x20]);
        assert!(deframe_c608(&lying).is_empty());
        let mut tiny = Vec::from(4u32.to_be_bytes());
        tiny.extend_from_slice(b"cdat");
        assert!(deframe_c608(&tiny).is_empty());
        let mut odd = Vec::from(11u32.to_be_bytes());
        odd.extend_from_slice(b"cdat");
        odd.extend_from_slice(&[0x94, 0x20, 0x00]);
        assert!(deframe_c608(&odd).is_empty());
        assert!(deframe_c608(&[0, 0, 0]).is_empty());
    }

    #[test]
    fn deframe_c708_unwraps_a_ccdp_packet() {
        use crate::mp4box::mp4_box;
        let triples = alloc::vec![
            CcTriple {
                cc_type: 3,
                b0: 0x21,
                b1: 0x40
            },
            CcTriple {
                cc_type: 2,
                b0: 0x41,
                b1: 0x42
            },
        ];
        let sample = mp4_box(b"ccdp", &crate::cea::build_cdp(&triples, 4, 7));
        assert_eq!(
            crate::cea::parse_cc_data(&deframe_c708(&sample)),
            triples,
            "the CDP's ccdata triples come back out"
        );
        // A CDP whose checksum is wrong contributes nothing, and a lying atom size
        // stops the scan rather than reading past the sample.
        let mut bad = mp4_box(b"ccdp", &crate::cea::build_cdp(&triples, 4, 7));
        *bad.last_mut().expect("checksum byte") ^= 0xFF;
        assert!(deframe_c708(&bad).is_empty());
        let mut lying = Vec::from(0x7FFF_FFFFu32.to_be_bytes());
        lying.extend_from_slice(b"ccdp");
        assert!(deframe_c708(&lying).is_empty());
    }

    #[test]
    fn deframe_wvtt_extracts_payl_payloads() {
        use crate::mp4box::mp4_box;
        // A vttc cue carries a payl payload; multiple cues in one sample join with
        // a newline. A vtte (empty cue) and unknown boxes contribute nothing.
        let cue = |id: &[u8], payl: &str| {
            let mut body = mp4_box(b"iden", id);
            body.extend_from_slice(&mp4_box(b"sttg", b"align:center"));
            body.extend_from_slice(&mp4_box(b"payl", payl.as_bytes()));
            mp4_box(b"vttc", &body)
        };
        let mut sample = cue(b"1", "Hello");
        sample.extend_from_slice(&cue(b"2", "World"));
        assert_eq!(deframe_wvtt(&sample), b"Hello\nWorld");

        // A pure empty-cue sample yields an empty string (clears the overlay).
        let empty = mp4_box(b"vtte", &[]);
        assert!(deframe_wvtt(&empty).is_empty());

        // A truncated box (size past the end) stops the scan, no panic.
        let truncated = [0, 0, 0, 0xFF, b'v', b't', b't', b'c'];
        assert!(deframe_wvtt(&truncated).is_empty());
    }

    #[test]
    fn fragments_read_version_1_headers_large_boxes_and_trex_defaults() {
        use crate::mp4box::{ftyp, full_box, mp4_box};
        const TRACK_ID: u32 = 1;
        const TIMESCALE: u32 = 90_000;
        const SAMPLE_TICKS: u32 = 3000;
        const LARGE_SIZE_MARKER: u32 = 1;
        let (width, height) = (320u32, 240u32);
        let sample: &[u8] = &[0, 0, 0, 2, 0x65, 0xAA];

        // version 1 tkhd / mdhd: 64-bit times and duration
        let mut tkhd = alloc::vec![0u8; 92];
        tkhd[16..20].copy_from_slice(&TRACK_ID.to_be_bytes());
        tkhd[84..88].copy_from_slice(&(width << 16).to_be_bytes());
        tkhd[88..92].copy_from_slice(&(height << 16).to_be_bytes());
        let mut mdhd = alloc::vec![0u8; 28];
        mdhd[16..20].copy_from_slice(&TIMESCALE.to_be_bytes());
        let mut hdlr = alloc::vec![0u8; 20];
        hdlr[4..8].copy_from_slice(b"vide");
        // one 2-byte SPS and one 2-byte PPS
        let avcc = [
            1, 0x42, 0, 0x1e, 0xFF, 0xE1, 0, 2, 0x67, 0x42, 1, 0, 2, 0x68, 0xce,
        ];
        let mut avc1 = alloc::vec![0u8; 78];
        avc1.extend_from_slice(&mp4_box(b"avcC", &avcc));
        let mut stsd = 1u32.to_be_bytes().to_vec();
        stsd.extend_from_slice(&mp4_box(b"avc1", &avc1));
        let stbl = mp4_box(b"stbl", &full_box(b"stsd", 0, 0, &stsd));
        let mdia = mp4_box(
            b"mdia",
            &[
                full_box(b"mdhd", 1, 0, &mdhd),
                full_box(b"hdlr", 0, 0, &hdlr),
                mp4_box(b"minf", &stbl),
            ]
            .concat(),
        );
        let trak = mp4_box(b"trak", &[full_box(b"tkhd", 1, 0, &tkhd), mdia].concat());
        let trex = full_box(
            b"trex",
            0,
            0,
            &[TRACK_ID, 1, SAMPLE_TICKS, sample.len() as u32, 0]
                .iter()
                .flat_map(|field| field.to_be_bytes())
                .collect::<Vec<u8>>(),
        );
        let moov = mp4_box(b"moov", &[trak, mp4_box(b"mvex", &trex)].concat());

        // no tfdt, and a trun that leaves duration and size to the trex
        let moof = mp4_box(
            b"moof",
            &mp4_box(
                b"traf",
                &[
                    full_box(b"tfhd", 0, 0, &TRACK_ID.to_be_bytes()),
                    full_box(b"trun", 0, 0, &1u32.to_be_bytes()),
                ]
                .concat(),
            ),
        );
        let mut large_mdat = LARGE_SIZE_MARKER.to_be_bytes().to_vec();
        large_mdat.extend_from_slice(b"mdat");
        large_mdat.extend_from_slice(&(16 + sample.len() as u64).to_be_bytes());
        large_mdat.extend_from_slice(sample);

        let mut file = ftyp();
        file.extend_from_slice(&moov);
        for _ in 0..2 {
            file.extend_from_slice(&moof);
            file.extend_from_slice(&large_mdat);
        }

        let tracks = parse_all_tracks(&file).expect("version 1 headers parse");
        assert_eq!(tracks[0].timescale, TIMESCALE);
        assert!(matches!(
            tracks[0].kind,
            TrackKind::Video { width: w, height: h, .. } if (w, h) == (width, height)
        ));
        let samples =
            parse_fragments_multi(&file, &tracks, 0, &mut FragmentStarts::default(), None)
                .expect("fragments parse");
        let decode_times: Vec<u64> = samples.iter().map(|(_, s)| s.dts_ns).collect();
        let sample_ns = ticks_to_ns(u64::from(SAMPLE_TICKS), TIMESCALE);
        assert_eq!(decode_times, alloc::vec![0, sample_ns]);
        assert_eq!(samples[1].1.duration_ns, sample_ns);
    }

    #[cfg(feature = "ffmpeg-long-tail")]
    #[test]
    fn an_unread_sample_entry_goes_on_under_its_fourcc() {
        const VISUAL_SAMPLE_ENTRY_LEN: usize = 78;
        let entry = crate::mp4box::mp4_box(b"rpza", &[0; VISUAL_SAMPLE_ENTRY_LEN]);
        let Some((
            TrackKind::Video {
                codec, param_sets, ..
            },
            _,
        )) = parse_video_entry(&entry, 128, 96).expect("the entry parses")
        else {
            panic!("rpza is read as a video track");
        };
        assert_eq!(
            codec,
            VideoCodec::Tagged(g2g_core::CodecTag {
                namespace: g2g_core::TagNamespace::Mov,
                tag: u32::from_le_bytes(*b"rpza"),
            })
        );
        assert_eq!(
            param_sets,
            [entry],
            "the whole sample entry leads the stream"
        );
    }

    #[cfg(feature = "ffmpeg-long-tail")]
    #[test]
    fn a_mace_track_is_cut_into_whole_frames_per_chunk() {
        const SOUND_ENTRY_V0_LEN: usize = 28;
        const CHANNELS: u16 = 2;
        const SAMPLE_RATE: u16 = 22_050;
        const CHUNK_SAMPLES: usize = 36;
        let mut payload = [0u8; SOUND_ENTRY_V0_LEN];
        payload[SOUND_ENTRY_CHANNELS].copy_from_slice(&CHANNELS.to_be_bytes());
        payload[SOUND_ENTRY_SAMPLE_RATE].copy_from_slice(&SAMPLE_RATE.to_be_bytes());
        let mac3 = u32::from_le_bytes(*b"MAC3");
        let sound = SoundDescription::parse(&payload, mac3).expect("the entry parses");
        // MACE 3:1 packs 6 samples per channel into 2 bytes
        assert_eq!((sound.samples_per_frame, sound.bytes_per_frame), (6, 4));
        let stored_bytes = CHUNK_SAMPLES / 6 * 4;
        let data: Vec<u8> = (0..2 * stored_bytes as u8).collect();
        let chunks = [(0u64, CHUNK_SAMPLES), (stored_bytes as u64, CHUNK_SAMPLES)];
        let samples = quicktime_chunk_samples(&data, &chunks, u32::from(SAMPLE_RATE), sound, 1);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[1].annexb, data[stored_bytes..]);
        assert_eq!(
            samples[1].pts_ns,
            ticks_to_ns(CHUNK_SAMPLES as u64, u32::from(SAMPLE_RATE))
        );
    }

    #[test]
    fn reads_a_2vuy_entry_as_raw_video_at_its_coded_size() {
        const VISUAL_SAMPLE_ENTRY_LEN: usize = 78;
        const CODED_WIDTH: u16 = 320;
        const CODED_HEIGHT: u16 = 240;
        let mut fields = [0u8; VISUAL_SAMPLE_ENTRY_LEN];
        fields[VISUAL_SAMPLE_ENTRY_WIDTH_AT..VISUAL_SAMPLE_ENTRY_WIDTH_AT + 2]
            .copy_from_slice(&CODED_WIDTH.to_be_bytes());
        fields[VISUAL_SAMPLE_ENTRY_HEIGHT_AT..VISUAL_SAMPLE_ENTRY_HEIGHT_AT + 2]
            .copy_from_slice(&CODED_HEIGHT.to_be_bytes());
        let entry = crate::mp4box::mp4_box(b"2vuy", &fields);
        let Some((kind, _)) = parse_video_entry(&entry, 640, 480).expect("the entry parses") else {
            panic!("2vuy is read as a video track");
        };
        let TrackKind::RawVideo {
            format,
            width,
            height,
            ..
        } = kind
        else {
            panic!("2vuy is uncompressed");
        };
        assert_eq!(format, RawVideoFormat::Uyvy);
        assert_eq!((width, height), (CODED_WIDTH.into(), CODED_HEIGHT.into()));
        let no_geometry = crate::mp4box::mp4_box(b"2vuy", &[0; VISUAL_SAMPLE_ENTRY_LEN]);
        assert!(parse_video_entry(&no_geometry, 640, 480).is_err());
    }

    #[test]
    fn reads_the_prores_and_dnxhd_sample_entries() {
        const VISUAL_SAMPLE_ENTRY_LEN: usize = 78;
        for (fourcc, want) in [
            (
                b"apch",
                VideoCodec::ProRes {
                    variant: ProResVariant::Hq,
                },
            ),
            (
                b"ap4h",
                VideoCodec::ProRes {
                    variant: ProResVariant::FourFourFourFour,
                },
            ),
            (b"AVdh", VideoCodec::DnxHd),
        ] {
            let entry = crate::mp4box::mp4_box(fourcc, &[0; VISUAL_SAMPLE_ENTRY_LEN]);
            let Some((TrackKind::Video { codec, .. }, _)) =
                parse_video_entry(&entry, 1920, 1080).expect("the entry parses")
            else {
                panic!("{fourcc:?} is read as a video track");
            };
            assert_eq!(codec, want);
        }
    }

    #[test]
    fn reads_avid_motion_jpeg_as_mjpeg() {
        const VISUAL_SAMPLE_ENTRY_LEN: usize = 78;
        let entry = crate::mp4box::mp4_box(b"AVDJ", &[0; VISUAL_SAMPLE_ENTRY_LEN]);
        let Some((TrackKind::Video { codec, .. }, _)) =
            parse_video_entry(&entry, 720, 486).expect("the entry parses")
        else {
            panic!("AVDJ is read as a video track");
        };
        assert_eq!(codec, VideoCodec::Mjpeg);
    }
}

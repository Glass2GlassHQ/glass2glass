//! Multi-track fragmented-MP4 multiplexer element (M293): N elementary streams
//! in (H.264 / H.265 video + AAC audio), one ISO-BMFF byte stream out. The A/V
//! analog of the single-track [`crate::mp4mux::Mp4Mux`], so a muxed recording
//! carries video and audio together:
//!
//! ```text
//! videotestsrc ! x264enc ! mp4mux name=m
//! audiotestsrc ! avenc_aac ! m.
//! m. ! filesink location=av.mp4
//! ```
//!
//! A [`MultiInputElement`] (input pad order = track order = `track_ID`): each pad
//! takes one elementary stream, and access units interleave by presentation
//! timestamp via the M204 [`InputAggregator`] merge before being written to their
//! track. The `moov` (one `trak` per stream) is built once every track has its
//! init data, which arrives in-band: a video track's parameter sets ride the
//! first IDR, an AAC track's AudioSpecificConfig is synthesised from the first
//! ADTS header (the AAC bytes are written de-ADTS'd into the `mdat`), an Opus
//! track's `dOps` comes from a leading `OpusHead` when one arrives. After the
//! init segment, one `moof`+`mdat` fragment per access unit, each `traf`
//! referencing its track with a per-track `tfdt` in that track's timescale.
//!
//! Reachable from the `gst-launch` fan-in syntax: registered as the `mp4mux`
//! muxer in `default_registry`, so >1 input link builds this element (a single
//! input builds the single-track [`crate::mp4mux::Mp4Mux`]), the way gst's
//! request sink pads do. Video is H.264/H.265 (avc1/hvc1), VP8/VP9
//! (vp08/vp09 + vpcC) or AV1 (av01 + av1C from the sequence header, M773);
//! audio is AAC (mp4a/esds) or Opus (Opus/dOps), sync-sample audio. A
//! `Caps::Text{Utf8}` pad adds a `tx3g` timed-text track (M898, ffmpeg's
//! `mov_text`): one cue per sample, with the runs between cues filled by empty
//! samples, since a text sample presents where the durations before it end.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    split_tags, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, Chapter,
    ClosedCaptionFormat, ConfigureOutcome, Dim, FrameTiming, G2gError, InputAggregator,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, TagList, TextFormat, VideoCodec,
};

use crate::cea::{build_cdp, parse_cc_data, CcTriple};
use crate::fmp4mux::{
    av1c_record, avcc_sample, is_keyframe_nal, parameter_sets, split_annexb, tfhd,
    visual_sample_entry, vp8_keyframe, vp9_keyframe,
};
use crate::mp4audiosink::esds;
use crate::mp4box::{ftyp, ftyp_cmaf, full_box, mp4_box, prft, styp_cmaf, udta_with_tags, MATRIX};
use crate::opusparse::{
    dops_from_opus_head, is_opus_config, parse_opus_head, OPUS_ENCODER_PRE_SKIP,
};
use crate::rtcp;

/// Video tracks use a 90 kHz media timescale; audio tracks use the sample rate.
const VIDEO_TIMESCALE: u32 = 90_000;
/// Timed-text tracks use a 1 kHz media timescale (ffmpeg's for `mov_text`), so a
/// cue's millisecond timing lands on a tick exactly.
const TEXT_TIMESCALE: u32 = 1_000;
/// CEA-708 frame-rate code for 29.97 fps, the rate a muxed CDP declares (the
/// North-American caption norm, and what `st2110ancrtp` defaults to).
const CDP_FRAME_RATE_2997: u8 = 4;
/// The `mvhd` timescale, so movie-level durations are milliseconds.
const MOVIE_TIMESCALE: u32 = 1000;
const DEFAULT_VIDEO_DURATION_NS: u64 = 33_333_333;

/// What an input pad carries, learned from its negotiated caps at configure.
#[derive(Debug, Clone, Copy)]
enum PadKind {
    Video(VideoCodec),
    Audio {
        format: AudioFormat,
        channels: u8,
        rate: u32,
    },
    /// A raw closed-caption track (M883): the pad carries `cc_data` triples, which
    /// go into `c608` / `c708` samples.
    ClosedCaption(ClosedCaptionFormat),
    /// A timed-text subtitle track (M898): the pad carries one plain-UTF-8 cue per
    /// frame, which goes into a `tx3g` (3GPP timed text, ffmpeg's `mov_text`)
    /// sample.
    Text,
}

/// A track's `moov` init data, captured from its first access unit.
#[derive(Debug, Clone)]
enum TrackInit {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
        param_sets: Vec<Vec<u8>>,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        rate: u32,
        /// The elementary stream's out-of-band config: the AAC
        /// AudioSpecificConfig synthesised from the first ADTS header, or an
        /// in-band Opus `OpusHead` (M791). Empty when the stream carried none,
        /// in which case the sample entry falls back to the caps.
        config: Vec<u8>,
    },
    ClosedCaption {
        format: ClosedCaptionFormat,
    },
    /// A subtitle track needs nothing from the stream, so it is ready at configure.
    Text,
}

impl TrackInit {
    fn timescale(&self) -> u32 {
        match self {
            TrackInit::Video { .. } | TrackInit::ClosedCaption { .. } => VIDEO_TIMESCALE,
            TrackInit::Audio { rate, .. } => *rate,
            TrackInit::Text => TEXT_TIMESCALE,
        }
    }
}

/// Muxes N elementary streams into one ISO-BMFF byte stream, PTS-ordered.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::mp4muxn::Mp4MuxN;
///
/// let mux = Mp4MuxN::new(2)
///     .with_fragmented(true)
///     .with_fragment_duration_ms(2000);
/// ```
#[derive(Debug)]
pub struct Mp4MuxN {
    inputs: usize,
    /// Per-pad stream kind, learned at configure (the moov needs every track).
    kinds: Vec<Option<PadKind>>,
    /// Per-pad track init, captured from the first AU. Geometry comes from the
    /// caps; the codec config comes in-band from the first AU.
    inits: Vec<Option<TrackInit>>,
    /// Per-pad caps geometry (video width/height), recorded at configure.
    dims: Vec<(u32, u32)>,
    agg: InputAggregator<Frame>,
    /// Per-track accumulated decode time in that track's timescale (`tfdt`).
    decode_time: Vec<u64>,
    /// Per-track previous PTS (ns), for the sample-duration delta.
    prev_pts_ns: Vec<Option<u64>>,
    /// Per-track CDP sequence counter, for a `c708` track's caption packets.
    cdp_seq: Vec<u16>,
    /// Per-track end (ns) of the last text cue written, so the run before the next
    /// one can be filled with an empty sample (see the gap note in `emit_au`).
    text_end_ns: Vec<u64>,
    header_written: bool,
    /// Global moof sequence number (1-based, increasing across the movie).
    sequence: u64,
    emitted: u64,
    /// Target fragment duration in milliseconds (`0` = one `moof`+`mdat` fragment
    /// per access unit, the default). Batches a track's access units into a
    /// multi-sample fragment closed at the next sync sample once the target is
    /// reached, matching the single-track [`crate::mp4mux::Mp4Mux`].
    fragment_duration_ms: u64,
    /// Per-track fragment being accumulated in batched mode (empty in per-AU mode).
    pending: Vec<PendingFragment>,
    /// Whether to write the fragmented layout (`moof`+`mdat` per fragment, the
    /// default and the only streamable one). `false` selects the progressive
    /// layout: see [`with_fragmented`](Self::with_fragmented).
    fragmented: bool,
    /// Whether the progressive layout puts its `moov` ahead of the `mdat`
    /// (M824); see [`with_faststart`](Self::with_faststart).
    faststart: bool,
    /// CMAF conformance mode (M832); see [`with_cmaf`](Self::with_cmaf).
    cmaf: bool,
    /// Target CMAF chunk duration in milliseconds (`0` = no chunking); see
    /// [`with_chunk_duration_ms`](Self::with_chunk_duration_ms).
    chunk_duration_ms: u64,
    /// Whether each fragment is preceded by a `prft`; see
    /// [`with_prft`](Self::with_prft).
    write_prft: bool,
    /// Progressive mode's buffered samples, in the global PTS-merged order they
    /// were released (which is also the `mdat` byte order). Empty in the
    /// fragmented default.
    samples: Vec<ProgSample>,
    /// Whole-file metadata, written as the `moov`'s `udta/meta/ilst`.
    tags: TagList,
    /// Per-input metadata, written as that `trak`'s own `udta/meta/ilst` (M838).
    /// One (possibly empty) list per input pad.
    track_tags: Vec<TagList>,
    /// The table of contents, written as the `moov`'s `udta/chpl` (M1046).
    chapters: Vec<Chapter>,
}

/// One buffered sample of the progressive (moov-at-end) layout: its `mdat`
/// bytes plus everything the sample tables need. Times are in the track's own
/// media timescale.
#[derive(Debug, Clone)]
struct ProgSample {
    input: usize,
    bytes: Vec<u8>,
    duration: u32,
    is_sync: bool,
    dts: u64,
    /// `pts - dts`, converted from the nanosecond difference rather than from
    /// two independently rounded tick values, so a constant reorder delay stays
    /// constant instead of dithering by a tick.
    composition_offset: u32,
}

/// One buffered sample of an in-progress fragment (batched mode).
#[derive(Debug, Clone)]
struct PendingSample {
    bytes: Vec<u8>,
    /// Sample duration in the track's timescale.
    duration: u32,
    is_sync: bool,
}

/// One finished sample on its way into the file: its bytes and the timing the
/// sample tables (or the fragment's `trun`) record for it. `duration` is in the
/// track's media timescale.
#[derive(Debug, Clone)]
struct TimedSample {
    bytes: Vec<u8>,
    is_sync: bool,
    pts_ns: u64,
    dts_ns: u64,
    duration: u32,
}

/// A track's in-progress `moof`+`mdat` fragment: the samples buffered so far, the
/// decode time at the fragment's first sample (its `tfdt`), and the accumulated
/// media duration (track timescale) used to decide when the target is reached.
/// With chunking on (M859) the buffered samples and the `tfdt` are the open
/// chunk's, while `accum_ticks` still measures the whole fragment.
#[derive(Debug, Clone, Default)]
struct PendingFragment {
    samples: Vec<PendingSample>,
    base_decode_time: u64,
    /// PTS (ns) of the fragment's first sample, carried on the emitted byte frame.
    base_pts_ns: u64,
    accum_ticks: u64,
    /// Media duration of the open chunk only (track timescale).
    chunk_ticks: u64,
    /// Whether a chunk of this fragment has already been emitted, i.e. its `styp`
    /// and `prft` are written and the next chunk continues it.
    started: bool,
    /// Producer wall clock (NTP) sampled at the fragment's first sample, written
    /// in its `prft`.
    ntp: u64,
}

impl Mp4MuxN {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "Mp4MuxN needs at least one input");
        Self {
            inputs,
            kinds: alloc::vec![None; inputs],
            inits: alloc::vec![None; inputs],
            dims: alloc::vec![(0, 0); inputs],
            agg: InputAggregator::new(inputs),
            decode_time: alloc::vec![0; inputs],
            prev_pts_ns: alloc::vec![None; inputs],
            cdp_seq: alloc::vec![0; inputs],
            text_end_ns: alloc::vec![0; inputs],
            header_written: false,
            sequence: 0,
            emitted: 0,
            fragment_duration_ms: 0,
            pending: alloc::vec![PendingFragment::default(); inputs],
            fragmented: true,
            faststart: false,
            cmaf: false,
            chunk_duration_ms: 0,
            write_prft: false,
            samples: Vec::new(),
            tags: TagList::new(),
            track_tags: alloc::vec![TagList::new(); inputs],
            chapters: Vec::new(),
        }
    }

    /// Attach whole-file metadata, written as the `moov`'s own
    /// `udta/meta/ilst` (the [`crate::mp4mux::Mp4Mux`] builder for the
    /// multi-track muxer).
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Attach metadata scoped to one input pad's track, written as that `trak`'s
    /// own `udta/meta/ilst`, which ffmpeg and the g2g demuxer report on the
    /// elementary stream rather than the file. Out-of-range inputs are ignored.
    ///
    /// A tag every input carries identically moves up to the file level instead
    /// (`g2g_core::split_tags`), and a tag also set by [`with_tags`](Self::with_tags)
    /// is not repeated per track unless the value differs, in which case the
    /// track's value stands for that track.
    pub fn with_track_tags(mut self, input: usize, tags: TagList) -> Self {
        if input < self.inputs {
            self.track_tags[input] = tags;
        }
        self
    }

    /// Attach the table of contents, written as the `moov`'s
    /// `udta/chpl` Nero chapter list (M1046). Chapter times are stream-time
    /// nanoseconds. Builder only: a launch line has no syntax for a chapter
    /// list.
    pub fn with_chapters(mut self, chapters: Vec<Chapter>) -> Self {
        self.chapters = chapters;
        self
    }

    /// The tags the `moov` writes, split by the shared global / per-stream merge
    /// policy into the file's own list and one list per pad slot.
    fn moov_metadata(&self) -> MoovMetadata {
        let (global, per_track) = split_tags(&self.tags, &self.track_tags);
        MoovMetadata {
            global,
            per_track,
            chapters: self.chapters.clone(),
        }
    }

    /// Batch access units into fragments of at least `ms` milliseconds (`0` keeps
    /// one fragment per AU); see [`fragment_duration_ms`](Self::fragment_duration_ms).
    pub fn with_fragment_duration_ms(mut self, ms: u64) -> Self {
        self.fragment_duration_ms = ms;
        self
    }

    /// Split each fragment into CMAF chunks of at least `ms` milliseconds (M859),
    /// each its own `moof`+`mdat` emitted the moment it fills, so a low-latency
    /// player receives part of a fragment before the fragment is complete. `0`
    /// (the default) writes one `moof`+`mdat` per fragment. Inert unless the
    /// muxer batches (`fragment-duration` set, or `cmaf`), since per-AU mode has
    /// no fragment to subdivide.
    pub fn with_chunk_duration_ms(mut self, ms: u64) -> Self {
        self.chunk_duration_ms = ms;
        self
    }

    /// Write a `prft` ahead of each fragment (M859) mapping the fragment's first
    /// decode time to the producer's wall clock (NTP), which is what lets a
    /// player measure its end-to-end latency against a chunked live stream. The
    /// box names the fragment's own track.
    pub fn with_prft(mut self, write_prft: bool) -> Self {
        self.write_prft = write_prft;
        self
    }

    /// Choose the output layout (M793). `true` (the default) is the fragmented
    /// one: `ftyp`+`moov` up front, then a `moof`+`mdat` per fragment, which
    /// streams and survives truncation. `false` is the progressive one: `ftyp`,
    /// one `mdat` holding every sample, then a `moov` with real sample tables
    /// (`stts` / `stsz` / `stsc` / `stco` / `stss` / `ctts`) and real
    /// `mvhd`/`tkhd`/`mdhd` durations, which is what a reader needs to report an
    /// exact duration (`ffprobe` derives a fragmented file's from the sample
    /// durations it sums, so an edit list's trim never shows there).
    ///
    /// Progressive is a two-pass mode like `matroskamux`'s `seekable`: the
    /// `moov` cannot be written until every sample's size is known, so the whole
    /// file is held in memory and emitted at EOS. Memory grows with the
    /// recording; a live or long capture wants the fragmented default.
    pub fn with_fragmented(mut self, fragmented: bool) -> Self {
        self.fragmented = fragmented;
        self
    }

    /// Put the `moov` ahead of the media data (M824), the `qtmux faststart`
    /// layout: a reader has the whole index before it has read a byte of `mdat`,
    /// so playback over a network starts without seeking to the end of the file.
    ///
    /// Only the progressive layout has anything to move: a fragmented file
    /// already writes its `moov` before the first fragment. It costs no extra
    /// buffering either, since progressive already holds the movie until EOS;
    /// the `moov` is simply written first, with the chunk offsets shifted by its
    /// own size.
    pub fn with_faststart(mut self, faststart: bool) -> Self {
        self.faststart = faststart;
        self
    }

    /// Write a CMAF (ISO/IEC 23000-19) track file (M832): the `ftyp` carries the
    /// `cmfc` structural brand, each fragment opens a CMAF segment with its own
    /// `styp`, its `tfhd` states the sample description rather than inheriting it,
    /// and a fragment starts only at a sync sample, so `fragment-duration = 0`
    /// means one fragment per GOP rather than one per access unit.
    ///
    /// CMAF puts one media stream in a track file, so this muxer only accepts it
    /// with a single input pad, and only in the fragmented layout; either
    /// combination fails at `configure_pipeline` rather than writing a file that
    /// claims a conformance it does not have.
    pub fn with_cmaf(mut self, cmaf: bool) -> Self {
        self.cmaf = cmaf;
        self
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }

    fn pad_kind_for(caps: &Caps) -> Option<PadKind> {
        match caps {
            Caps::CompressedVideo {
                codec:
                    c @ (VideoCodec::H264
                    | VideoCodec::H265
                    | VideoCodec::Vp8
                    | VideoCodec::Vp9
                    | VideoCodec::Av1),
                ..
            } => Some(PadKind::Video(*c)),
            Caps::Audio {
                format: format @ (AudioFormat::Aac | AudioFormat::Opus),
                channels,
                sample_rate,
            } => Some(PadKind::Audio {
                format: *format,
                channels: *channels,
                rate: *sample_rate,
            }),
            Caps::ClosedCaption { format } => Some(PadKind::ClosedCaption(*format)),
            // Only the elementary cue form: a `Text` pad of a document format
            // (`Srt` / `Ssa` / `Ttml`) carries whole-file bytes, not timed cues.
            Caps::Text {
                format: TextFormat::Utf8,
            } => Some(PadKind::Text),
            _ => None,
        }
    }

    /// Adopt a concrete channel count / sample rate from a runtime
    /// `CapsChanged`, which is how a demuxer delivers them (its negotiation caps
    /// carry the `0/0` sentinel, so without this the audio `trak` would declare
    /// a zero `mdhd` timescale and no reader could time the track). Only fills
    /// in a field the pad does not have yet, and only while the `moov` is still
    /// unwritten; a mid-stream format change cannot be expressed in a `moov`
    /// already on the wire.
    fn refine_audio_caps(&mut self, input: usize, caps: &Caps) {
        let Some(PadKind::Audio {
            channels: new_channels,
            rate: new_rate,
            ..
        }) = Self::pad_kind_for(caps)
        else {
            return;
        };
        if self.header_written {
            return;
        }
        let fill = |channels: &mut u8, rate: &mut u32| {
            if *channels == 0 {
                *channels = new_channels;
            }
            if *rate == 0 {
                *rate = new_rate;
            }
        };
        if let Some(PadKind::Audio { channels, rate, .. }) = &mut self.kinds[input] {
            fill(channels, rate);
        }
        if let Some(TrackInit::Audio { channels, rate, .. }) = &mut self.inits[input] {
            fill(channels, rate);
        }
    }

    /// Whether the pad carries Opus, whose in-band headers are dropped rather
    /// than muxed as samples.
    fn is_opus_pad(&self, input: usize) -> bool {
        matches!(
            self.kinds[input],
            Some(PadKind::Audio {
                format: AudioFormat::Opus,
                ..
            })
        )
    }

    /// True once every pad that will produce data has its init captured. A pad
    /// that ended without an AU is excluded (its track is simply absent).
    fn all_inits_ready(&self) -> bool {
        (0..self.inputs).all(|i| self.inits[i].is_some() || self.agg.is_ended(i))
    }

    /// Capture a pad's track init from its first access unit, if not already set.
    fn capture_init(&mut self, input: usize, au: &[u8]) {
        if self.inits[input].is_some() {
            return;
        }
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => {
                let (w, h) = self.dims[input];
                match codec {
                    VideoCodec::H264 | VideoCodec::H265 => {
                        let nalus = split_annexb(au);
                        // Parameter sets only ride the IDR; a leading P-frame has
                        // none, so wait for the keyframe that carries them.
                        if let Ok(param_sets) = parameter_sets(codec, &nalus) {
                            let owned: Vec<Vec<u8>> =
                                param_sets.iter().map(|s| s.to_vec()).collect();
                            self.inits[input] = Some(TrackInit::Video {
                                codec,
                                width: w,
                                height: h,
                                param_sets: owned,
                            });
                        }
                    }
                    // AV1's `av1C` needs the sequence-header OBU; wait for the
                    // temporal unit that carries it (keyframes do).
                    VideoCodec::Av1 => {
                        if let Some((_, obu)) = crate::av1parse::seq_header_obu(au) {
                            self.inits[input] = Some(TrackInit::Video {
                                codec,
                                width: w,
                                height: h,
                                param_sets: alloc::vec![obu.to_vec()],
                            });
                        }
                    }
                    // VP8/VP9 carry no out-of-band parameter sets; the track is
                    // ready at the first frame (the vpcC uses caps geometry).
                    _ => {
                        self.inits[input] = Some(TrackInit::Video {
                            codec,
                            width: w,
                            height: h,
                            param_sets: Vec::new(),
                        });
                    }
                }
            }
            Some(PadKind::Audio {
                format,
                channels,
                rate,
            }) => match format {
                // AAC's AudioSpecificConfig is synthesised from the first ADTS header.
                AudioFormat::Aac => {
                    if let Some(asc) = asc_from_adts(au) {
                        self.inits[input] = Some(TrackInit::Audio {
                            format,
                            channels,
                            rate,
                            config: asc.to_vec(),
                        });
                    }
                }
                // An Opus stream leads with its `OpusHead` when it came from a
                // container (M791), and the `dOps` is built from it; a freshly
                // encoded one has none, so the track is ready at the first packet
                // and the `dOps` falls back to the caps + libopus' lookahead.
                _ => {
                    let config = match parse_opus_head(au) {
                        Some(_) => au.to_vec(),
                        // An `OpusTags` ahead of the identification header is not
                        // the config: wait rather than fix an empty one.
                        None if is_opus_config(au) => return,
                        None => Vec::new(),
                    };
                    self.inits[input] = Some(TrackInit::Audio {
                        format,
                        channels,
                        rate,
                        config,
                    });
                }
            },
            // A caption track needs no out-of-band config: it is ready at its
            // first sample.
            Some(PadKind::ClosedCaption(format)) => {
                self.inits[input] = Some(TrackInit::ClosedCaption { format });
            }
            // A text track's init was fixed at configure; nothing rides the cues.
            Some(PadKind::Text) | None => {}
        }
    }

    /// The mdat sample bytes for a track: AVCC length-prefixed NALUs for video,
    /// the de-ADTS'd raw AAC for audio, and the caption atoms (`cdat` / `cdt2` for
    /// 608, `ccdp` for 708) for a raw-caption track. Also returns whether it is a
    /// sync sample.
    fn sample_for(&mut self, input: usize, au: &[u8]) -> (Vec<u8>, bool) {
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => match codec {
                VideoCodec::H264 | VideoCodec::H265 => {
                    let nalus = split_annexb(au);
                    let is_sync = nalus.iter().any(|n| is_keyframe_nal(codec, n));
                    (avcc_sample(&nalus), is_sync)
                }
                // VP8/VP9 frames are stored verbatim; keyframe from the frame header.
                VideoCodec::Vp8 => (au.to_vec(), vp8_keyframe(au)),
                VideoCodec::Av1 => (
                    crate::av1parse::strip_temporal_delimiters(au),
                    crate::av1parse::av1_keyframe(au),
                ),
                _ => (au.to_vec(), vp9_keyframe(au)),
            },
            // Audio access units are always sync samples. AAC strips its ADTS
            // header; Opus packets are stored raw.
            Some(PadKind::Audio {
                format: AudioFormat::Aac,
                ..
            }) => (strip_adts(au).to_vec(), true),
            // A caption frame is a cc_data triple stream; re-frame it into the
            // sample atoms the QuickTime caption entry defines.
            Some(PadKind::ClosedCaption(format)) => {
                let seq = self.cdp_seq[input];
                self.cdp_seq[input] = seq.wrapping_add(1);
                (cc_sample(au, format, seq), true)
            }
            Some(PadKind::Text) => (tx3g_sample(au), true),
            _ => (au.to_vec(), true),
        }
    }

    /// Buffer one access unit for its track. In per-AU mode (`fragment-duration`
    /// = 0) it is flushed immediately as its own `moof`+`mdat` fragment; in
    /// batched mode it accumulates into the track's pending fragment, which is
    /// closed at the next sync sample once the target duration is reached.
    async fn emit_au(
        &mut self,
        input: usize,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let au = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let pts_ns = frame.timing.pts_ns;
        let (sample, is_sync) = self.sample_for(input, au);

        let track = &self.inits[input];
        let timescale = track
            .as_ref()
            .map(TrackInit::timescale)
            .unwrap_or(VIDEO_TIMESCALE);
        let default_dur_ns = match self.kinds[input] {
            // An Opus packet's TOC states its own duration (2.5..60 ms), so read
            // it rather than assuming the 20 ms default; AAC frames 1024 samples.
            Some(PadKind::Audio {
                format: AudioFormat::Opus,
                rate,
                ..
            }) => {
                let samples = u64::from(crate::opusparse::packet_samples(au)).max(1);
                samples * 1_000_000_000 / rate.max(1) as u64
            }
            Some(PadKind::Audio { rate, .. }) => 1024 * 1_000_000_000 / rate.max(1) as u64,
            _ => DEFAULT_VIDEO_DURATION_NS,
        };
        let first_sample = self.prev_pts_ns[input].is_none();
        // The frame's own duration when upstream timed it (a demuxer knows the
        // container's end-of-stream trim, which no PTS delta shows), else the
        // delta from the previous sample, else the codec's nominal frame length.
        let dur_ns = match (frame.timing.duration_ns, self.prev_pts_ns[input]) {
            (d, _) if d > 0 => d,
            (_, Some(prev)) if pts_ns > prev => pts_ns - prev,
            _ => default_dur_ns,
        };
        self.prev_pts_ns[input] = Some(pts_ns);
        let duration = ns_to_ts(dur_ns, timescale) as u32;

        // A text track has no per-sample timestamp on disk: a cue presents where
        // the durations before it end, so the run between two cues (and any before
        // the first) must be filled with an empty sample, or every cue after a gap
        // shows early. This is what "no subtitle on screen" is in a tx3g track, and
        // what ffmpeg writes.
        if matches!(self.kinds[input], Some(PadKind::Text)) {
            let end_ns = self.text_end_ns[input];
            let gap = ns_to_ts(pts_ns.saturating_sub(end_ns), timescale) as u32;
            if gap > 0 {
                let filler = TimedSample {
                    bytes: tx3g_sample(&[]),
                    is_sync: true,
                    pts_ns: end_ns,
                    dts_ns: end_ns,
                    duration: gap,
                };
                self.push_sample(input, filler, timescale, out).await?;
            }
            self.text_end_ns[input] = pts_ns.saturating_add(dur_ns);
        }

        // A zero `dts_ns` means upstream is not timing decode order, except on a
        // track's first sample, where it is also where a reordered stream really
        // starts. A DTS past the PTS is not expressible in `ctts` version 0, so
        // such a sample decodes when it presents.
        let dts_ns = match frame.timing.dts_ns {
            d if d <= pts_ns && (d > 0 || first_sample) => d,
            _ => pts_ns,
        };
        let timed = TimedSample {
            bytes: sample,
            is_sync,
            pts_ns,
            dts_ns,
            duration,
        };
        self.push_sample(input, timed, timescale, out).await
    }

    /// Buffer one finished sample for its track: into the progressive file's
    /// sample list, or into the fragmented layout's pending fragment (flushed
    /// immediately in per-AU mode). Times are the sample's own, so a synthesised
    /// one (a text gap) goes in the same way a real access unit does.
    async fn push_sample(
        &mut self,
        input: usize,
        sample: TimedSample,
        timescale: u32,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let TimedSample {
            bytes: sample,
            is_sync,
            pts_ns,
            dts_ns,
            duration,
        } = sample;
        // Progressive mode: hold every sample until EOS, where the `mdat` and the
        // sample tables are written together. A frame with no decode timestamp of
        // its own (or one past its PTS, which `ctts` version 0 cannot express)
        // decodes when it presents, the common no-reorder case.
        if !self.fragmented {
            self.samples.push(ProgSample {
                input,
                bytes: sample,
                duration,
                is_sync,
                dts: ns_to_ts(dts_ns, timescale),
                composition_offset: ns_to_ts(pts_ns - dts_ns, timescale) as u32,
            });
            return Ok(());
        }

        // Batched mode closes the open fragment before starting a new one at a sync
        // sample once the target duration is reached, so every fragment begins on a
        // keyframe (audio access units are all sync, so they close on the target).
        // CMAF mode batches even with no target, because a fragment there may only
        // start at a sync sample.
        let target = self.frag_target_ticks(input, timescale);
        // With chunking on the buffered samples are only the open chunk, so a
        // fragment is open while any of its chunks has been emitted too.
        let open = self.pending[input].started || !self.pending[input].samples.is_empty();
        if (target > 0 || self.cmaf) && is_sync && open && self.pending[input].accum_ticks >= target
        {
            self.flush_track(input, out).await?;
            self.close_fragment(input);
        }

        let write_prft = self.write_prft;
        let decode_time = self.decode_time[input];
        let pend = &mut self.pending[input];
        if pend.samples.is_empty() {
            pend.base_decode_time = decode_time;
            pend.base_pts_ns = pts_ns;
            if !pend.started {
                // the producer's wall clock at the fragment's first sample is what
                // its prft maps to that sample's decode time.
                pend.ntp = if write_prft { rtcp::ntp_now() } else { 0 };
            }
        }
        pend.samples.push(PendingSample {
            bytes: sample,
            duration,
            is_sync,
        });
        pend.accum_ticks += duration as u64;
        pend.chunk_ticks += duration as u64;
        self.decode_time[input] += duration as u64;

        // Per-AU mode (target 0): flush immediately, one fragment per access unit.
        if target == 0 && !self.cmaf {
            self.flush_track(input, out).await?;
            self.close_fragment(input);
            return Ok(());
        }
        // Close the chunk as soon as it reaches its target: the point of chunking
        // is that these bytes leave now rather than at the end of the fragment.
        let chunk_target = self.chunk_target_ticks(timescale);
        if chunk_target > 0 && self.pending[input].chunk_ticks >= chunk_target {
            self.flush_track(input, out).await?;
        }
        Ok(())
    }

    /// The fragment-duration target in a track's timescale (`0` = per-AU mode).
    fn frag_target_ticks(&self, _input: usize, timescale: u32) -> u64 {
        if self.fragment_duration_ms == 0 {
            return 0;
        }
        ns_to_ts(
            self.fragment_duration_ms.saturating_mul(1_000_000),
            timescale,
        )
    }

    /// The chunk-duration target in a track's timescale (`0` = no chunking).
    fn chunk_target_ticks(&self, timescale: u32) -> u64 {
        if self.chunk_duration_ms == 0 {
            return 0;
        }
        ns_to_ts(self.chunk_duration_ms.saturating_mul(1_000_000), timescale)
    }

    /// End the open fragment on a track: the next chunk written there opens a new
    /// one, with its own `styp` and `prft`.
    fn close_fragment(&mut self, input: usize) {
        let pend = &mut self.pending[input];
        pend.accum_ticks = 0;
        pend.started = false;
    }

    /// Write a track's buffered samples as one `moof`+`mdat` (a multi-sample
    /// `trun`), prepending the `ftyp`+`moov` init segment on the first fragment.
    /// That is the whole fragment unless chunking is on, in which case it is the
    /// open chunk and only the one that opens a fragment carries its `styp` and
    /// `prft`. A no-op when the track has no buffered samples.
    async fn flush_track(
        &mut self,
        input: usize,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if self.pending[input].samples.is_empty() {
            return Ok(());
        }
        let samples = core::mem::take(&mut self.pending[input].samples);
        let pend = &mut self.pending[input];
        let (base_decode_time, base_pts_ns, opens_fragment, ntp) = (
            pend.base_decode_time,
            pend.base_pts_ns,
            !pend.started,
            pend.ntp,
        );
        pend.chunk_ticks = 0;
        pend.started = true;

        // track_ID is the 1-based pad index; PTS of the first buffered sample.
        let track_id = (input + 1) as u32;
        let mut bytes = Vec::new();
        if !self.header_written {
            bytes.extend_from_slice(&if self.cmaf { ftyp_cmaf() } else { ftyp() });
            bytes.extend_from_slice(&av_moov(&self.inits, None, &self.moov_metadata()));
            self.header_written = true;
        }
        if opens_fragment {
            if self.cmaf {
                // this fragment is a CMAF segment, of chunks when chunking is on
                bytes.extend_from_slice(&styp_cmaf(self.chunk_duration_ms > 0));
            }
            if self.write_prft {
                bytes.extend_from_slice(&prft(track_id, ntp, base_decode_time));
            }
        }

        self.sequence += 1;
        bytes.extend_from_slice(&av_fragment(
            track_id,
            self.sequence,
            base_decode_time,
            self.cmaf,
            &samples,
        ));

        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns: base_pts_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }

    /// Flush every track's pending fragment (batched mode, at EOS), in track order.
    async fn flush_all(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        for input in 0..self.inputs {
            self.flush_track(input, out).await?;
            self.close_fragment(input);
        }
        Ok(())
    }

    /// Progressive mode's finalize (M793): write `ftyp` + one `mdat` holding
    /// every buffered sample + a `moov` whose sample tables index them, and emit
    /// the whole file as one frame. With `faststart` (M824) the `moov` goes
    /// between the `ftyp` and the `mdat` instead. A no-op when nothing was
    /// buffered.
    async fn finish_progressive(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.samples.is_empty() {
            return Ok(());
        }
        let samples = core::mem::take(&mut self.samples);
        let head = ftyp();

        // The mdat payload is the samples in the order they were released, so
        // each sample's offset inside the mdat is known once its header size is.
        let payload: usize = samples.iter().map(|s| s.bytes.len()).sum();
        // 32-bit box size unless the payload needs the 64-bit `largesize` form.
        let mdat_header = if payload.saturating_add(8) > u32::MAX as usize {
            16
        } else {
            8
        };
        let mut within = Vec::with_capacity(samples.len());
        let mut mdat_len = mdat_header as u64;
        for s in &samples {
            within.push(mdat_len);
            mdat_len = mdat_len.saturating_add(s.bytes.len() as u64);
        }

        let metadata = self.moov_metadata();
        // The sample tables address the mdat by absolute file offset, so the
        // moov's contents depend on where the mdat lands. `force_co64` pins the
        // chunk-offset entry width, which is the only thing that can change the
        // moov's own size once the sample count is fixed.
        let moov_at = |mdat_start: u64, force_co64: bool| -> Vec<u8> {
            let offsets: Vec<u64> = within
                .iter()
                .map(|w| mdat_start.saturating_add(*w))
                .collect();
            let tables: Vec<Option<TrackTables>> = (0..self.inputs)
                .map(|input| {
                    self.inits[input]
                        .as_ref()
                        .map(|init| track_tables(input, init, &samples, &offsets, force_co64))
                })
                .collect();
            av_moov(&self.inits, Some(&tables), &metadata)
        };

        let head_len = head.len() as u64;
        let (moov, mdat_start) = if self.faststart {
            // moov first: its size shifts the mdat, which changes the offsets the
            // moov itself stores. Pin the entry width from an upper bound on the
            // final offsets (the moov measured with the wider `co64` form is at
            // least as long as the real one), after which the moov built against
            // any mdat start has the size the real one will have.
            let widest = moov_at(0, true).len() as u64;
            let last_offset = head_len.saturating_add(widest).saturating_add(mdat_len);
            let force_co64 = last_offset > u64::from(u32::MAX);
            let moov_len = moov_at(0, force_co64).len() as u64;
            let start = head_len.saturating_add(moov_len);
            (moov_at(start, force_co64), start)
        } else {
            (moov_at(head_len, false), head_len)
        };

        let mut file = Vec::with_capacity(head.len() + moov.len() + mdat_len as usize);
        file.extend_from_slice(&head);
        if self.faststart {
            file.extend_from_slice(&moov);
        }
        debug_assert_eq!(
            file.len() as u64,
            mdat_start,
            "the mdat lands where the sample tables address it"
        );
        if mdat_header == 16 {
            file.extend_from_slice(&1u32.to_be_bytes()); // size 1: largesize follows
            file.extend_from_slice(b"mdat");
            file.extend_from_slice(&((payload + 16) as u64).to_be_bytes());
        } else {
            file.extend_from_slice(&((payload + 8) as u32).to_be_bytes());
            file.extend_from_slice(b"mdat");
        }
        for s in &samples {
            file.extend_from_slice(&s.bytes);
        }
        if !self.faststart {
            file.extend_from_slice(&moov);
        }

        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
            FrameTiming::default(),
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

impl MultiInputElement for Mp4MuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so every pad takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads (M481): a container mux's inputs are caps-typed slots, so
    /// `video_%u` / `audio_%u` / `sink_%u` each claim the next positional slot (the
    /// track type is read from the input's caps, not its index), so a launch line
    /// can name the pads (`m.video_0` / `m.audio_0`) in any order.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::pad_kind_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps_value(),
        )))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        // CMAF puts exactly one media stream in a track file, in the fragmented
        // layout. Refuse rather than write a file that claims a brand it breaks.
        if self.cmaf && (self.inputs > 1 || !self.fragmented) {
            return Err(G2gError::CapsMismatch);
        }
        let kind = Self::pad_kind_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if let Caps::CompressedVideo {
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        {
            self.dims[input] = (*w, *h);
        }
        // A text track's `trak` needs nothing from the stream, so it is ready now:
        // the first cue can be many seconds in, and the `moov` (which waits on
        // every track) would hold the A/V until then.
        if matches!(kind, PadKind::Text) {
            self.inits[input] = Some(TrackInit::Text);
        }
        self.kinds[input] = Some(kind);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
    }

    fn properties(&self) -> &'static [PropertySpec] {
        // `fragmented` has no GStreamer counterpart: `mp4mux` switches layout
        // with `fragment-duration = 0`, which g2g already spends on "one
        // fragment per access unit" (the streaming default). Reusing it would
        // flip the default output, so the layout gets its own boolean.
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "fragment-duration",
                PropKind::Uint,
                "target fragment duration, milliseconds (0 = one fragment per access unit)",
            )
            .with_default("0"),
            PropertySpec::new(
                "fragmented",
                PropKind::Bool,
                "moof/mdat fragments (streamable); false buffers the file and writes one mdat + a moov with real sample tables at EOS",
            )
            .with_default("true"),
            PropertySpec::new(
                "faststart",
                PropKind::Bool,
                "write the moov ahead of the mdat (progressive layout; a fragmented file's moov already leads)",
            )
            .with_default("false"),
            PropertySpec::new(
                "cmaf",
                PropKind::Bool,
                "write a CMAF track file: cmfc brands, a styp per segment, and fragments starting only at a sync sample (one input pad, fragmented layout)",
            )
            .with_default("false"),
            PropertySpec::new(
                "chunk-duration",
                PropKind::Uint,
                "target CMAF chunk duration, milliseconds (0 = one moof+mdat per fragment); a chunk is emitted as soon as it fills",
            )
            .with_default("0"),
            PropertySpec::new(
                "write-prft",
                PropKind::Bool,
                "write a producer reference time box (prft) ahead of each fragment, mapping its decode time to the wall clock",
            )
            .with_default("false"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "fragment-duration" => {
                self.fragment_duration_ms = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "fragmented" => {
                self.fragmented = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "faststart" => {
                self.faststart = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            // A CMAF track file holds one media stream, so refuse the mode on a
            // fan-in muxer at the point it is asked for.
            "cmaf" => {
                let on = value.as_bool().ok_or(PropError::Type)?;
                if on && self.inputs > 1 {
                    return Err(PropError::Value);
                }
                self.cmaf = on;
                Ok(())
            }
            "chunk-duration" => {
                self.chunk_duration_ms = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "write-prft" => {
                self.write_prft = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "fragment-duration" => Some(PropValue::Uint(self.fragment_duration_ms)),
            "fragmented" => Some(PropValue::Bool(self.fragmented)),
            "faststart" => Some(PropValue::Bool(self.faststart)),
            "cmaf" => Some(PropValue::Bool(self.cmaf)),
            "chunk-duration" => Some(PropValue::Uint(self.chunk_duration_ms)),
            "write-prft" => Some(PropValue::Bool(self.write_prft)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // Capture this track's init from its first AU before queueing.
                    if let Some(s) = frame.domain.as_system_slice() {
                        self.capture_init(input, s);
                        // An in-band Opus header is codec config, not audio: it
                        // built the `dOps` above and must never reach the `mdat`.
                        if self.is_opus_pad(input) && is_opus_config(s) {
                            return Ok(());
                        }
                    }
                    self.agg.push(input, frame);
                }
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // A demuxer negotiates compressed audio with the `0/0` "unknown
                // until parsed" caps and refines them at runtime, so the real
                // channel count / rate arrive here. Take them while the `moov`
                // can still change; otherwise the packet is the runner's to
                // consume (the moov is fixed from the first AU's in-band init).
                PipelinePacket::CapsChanged(caps) => {
                    self.refine_audio_caps(input, &caps);
                    return Ok(());
                }
                // A per-input `Segment` maps that stream to running time; a muxed
                // container carries its own timestamps, so it is consumed rather
                // than forwarded into the byte stream.
                PipelinePacket::Segment(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // Hold every AU until all tracks have their init (the moov needs them).
            if !self.all_inits_ready() {
                return Ok(());
            }
            // Release AUs now safe to emit, in global PTS order.
            while let Some((track, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                self.emit_au(track, frame, out).await?;
            }
            // At EOS (every input ended and drained), close any open batched
            // fragments so the last AUs are written; a no-op in per-AU mode.
            // Progressive mode instead writes the whole file now, since the
            // sample tables need every sample's size and offset.
            if self.agg.is_drained() {
                if self.fragmented {
                    self.flush_all(out).await?;
                } else {
                    self.finish_progressive(out).await?;
                }
            }
            Ok(())
        })
    }
}

// The ADTS de-frame / ASC-synthesis pair moved to the ungated `aacparse`
// module (M662, the no_std FLV muxer shares them); re-exported so this
// module's users keep their import path.
pub(crate) use crate::aacparse::{asc_from_adts, strip_adts};

fn ns_to_ts(ns: u64, timescale: u32) -> u64 {
    (ns as u128 * timescale as u128 / 1_000_000_000) as u64
}

/// The metadata a `moov` carries: the file's own tags and one list per pad slot,
/// already split by the global / per-stream merge policy (M838), plus the
/// file's table of contents (M1046, which has no per-track scope).
#[derive(Debug, Default)]
struct MoovMetadata {
    global: TagList,
    per_track: Vec<TagList>,
    chapters: Vec<Chapter>,
}

/// Build a multi-track `moov`: `mvhd` + one `trak` per track + `mvex` (one
/// `trex` per track) + the file's `udta` metadata. `track_ID` is the 1-based pad
/// slot, so it matches the `track_ID` each fragment carries even when a pad slot
/// is empty (no AU yet).
///
/// `tables` is `None` for the fragmented layout (empty sample tables, zero
/// durations, `mvex` declares the movie extends into fragments) and `Some` for
/// the progressive one (real tables and durations, no `mvex`), one entry per
/// pad slot in the same order as `tracks`.
fn av_moov(
    tracks: &[Option<TrackInit>],
    tables: Option<&[Option<TrackTables>]>,
    metadata: &MoovMetadata,
) -> Vec<u8> {
    let table_of = |i: usize| tables.and_then(|t| t.get(i)).and_then(Option::as_ref);
    let next_track_id = (tracks.len() + 1) as u32;
    // The movie lasts as long as its longest track's presentation.
    let movie_duration = (0..tracks.len())
        .filter_map(|i| table_of(i).map(|t| t.track_duration))
        .max()
        .unwrap_or(0);
    let mvhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
        p.extend_from_slice(&(movie_duration as u32).to_be_bytes()); // 0 when fragmented
        p.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate 1.0
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        p.extend_from_slice(&[0u8; 10]);
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&[0u8; 24]);
        p.extend_from_slice(&next_track_id.to_be_bytes());
        full_box(b"mvhd", 0, 0, &p)
    };

    let mut body = mvhd;
    for (i, track) in tracks.iter().enumerate() {
        let Some(track) = track else { continue };
        let track_tags = metadata.per_track.get(i);
        body.extend_from_slice(&trak(i as u32 + 1, track, table_of(i), track_tags));
    }
    // A progressive file has no fragments, so no `mvex` to announce them.
    if tables.is_none() {
        let mut p = Vec::new();
        for (i, track) in tracks.iter().enumerate() {
            if track.is_none() {
                continue;
            }
            let mut t = Vec::new();
            t.extend_from_slice(&(i as u32 + 1).to_be_bytes()); // track id
            t.extend_from_slice(&1u32.to_be_bytes()); // default sample description
            t.extend_from_slice(&[0u8; 12]); // default duration/size/flags
            p.extend_from_slice(&full_box(b"trex", 0, 0, &t));
        }
        body.extend_from_slice(&mp4_box(b"mvex", &p));
    }
    if let Some(udta) = udta_with_tags(&metadata.global, &metadata.chapters) {
        body.extend_from_slice(&udta);
    }
    mp4_box(b"moov", &body)
}

/// One progressive track's finished sample tables and the durations its headers
/// declare. Built by [`track_tables`] once every sample is buffered.
#[derive(Debug)]
struct TrackTables {
    /// `stts` + optional `ctts` + optional `stss` + `stsc` + `stsz` +
    /// `stco`/`co64`, in the order they go into the `stbl` after the `stsd`.
    tables: Vec<u8>,
    /// Total decode duration in the track's media timescale (`mdhd`).
    media_duration: u64,
    /// Presentation duration in the movie timescale (`tkhd`, and the edit list's
    /// `segment_duration`): the media duration less any codec delay trimmed by
    /// the edit.
    track_duration: u64,
}

/// Build one progressive track's sample tables from the muxer's buffered
/// samples and their `mdat` file offsets (`offsets[i]` locates `samples[i]`).
///
/// Each sample is its own chunk, so `stsc` is a single entry and `stco` has one
/// offset per sample: the samples of the tracks interleave in the `mdat`, so
/// larger chunks would need a chunking policy, and the 4 bytes per sample are
/// nothing next to the sample bytes already held in memory. Tables are ordered
/// by decode timestamp (a stable sort, so a stream with no reorder keeps the
/// order it arrived in), and `ctts` carries `pts - dts` where they differ.
///
/// `force_co64` writes 64-bit chunk offsets even when the current ones fit in
/// 32 bits: the faststart layout sizes the `moov` before it knows the offsets it
/// will hold, so it needs the width pinned rather than derived.
fn track_tables(
    input: usize,
    init: &TrackInit,
    samples: &[ProgSample],
    offsets: &[u64],
    force_co64: bool,
) -> TrackTables {
    let mut idx: Vec<usize> = (0..samples.len())
        .filter(|&i| samples[i].input == input)
        .collect();
    idx.sort_by_key(|&i| samples[i].dts);

    // stts: (count, delta) runs over the decode durations.
    let mut stts_runs: Vec<(u32, u32)> = Vec::new();
    let mut media_duration = 0u64;
    for &i in &idx {
        let d = samples[i].duration;
        media_duration += u64::from(d);
        match stts_runs.last_mut() {
            Some((n, delta)) if *delta == d => *n += 1,
            _ => stts_runs.push((1, d)),
        }
    }
    let mut stts = Vec::new();
    stts.extend_from_slice(&(stts_runs.len() as u32).to_be_bytes());
    for (n, delta) in &stts_runs {
        stts.extend_from_slice(&n.to_be_bytes());
        stts.extend_from_slice(&delta.to_be_bytes());
    }

    // ctts: composition offsets, run-length encoded, omitted when every sample
    // presents when it decodes (version 0, unsigned, which is what ffmpeg
    // writes unless asked for negative offsets).
    let mut ctts_runs: Vec<(u32, u32)> = Vec::new();
    for &i in &idx {
        let off = samples[i].composition_offset;
        match ctts_runs.last_mut() {
            Some((n, o)) if *o == off => *n += 1,
            _ => ctts_runs.push((1, off)),
        }
    }
    let ctts = if ctts_runs.iter().all(|(_, o)| *o == 0) {
        Vec::new()
    } else {
        let mut p = Vec::new();
        p.extend_from_slice(&(ctts_runs.len() as u32).to_be_bytes());
        for (n, off) in &ctts_runs {
            p.extend_from_slice(&n.to_be_bytes());
            p.extend_from_slice(&off.to_be_bytes());
        }
        full_box(b"ctts", 0, 0, &p)
    };

    // stss: 1-based sync sample numbers, omitted when every sample is one
    // (audio, and video with no inter-coded frames), which is what its absence
    // means.
    let syncs: Vec<u32> = idx
        .iter()
        .enumerate()
        .filter(|(_, &i)| samples[i].is_sync)
        .map(|(n, _)| n as u32 + 1)
        .collect();
    let stss = if syncs.len() == idx.len() {
        Vec::new()
    } else {
        let mut p = Vec::new();
        p.extend_from_slice(&(syncs.len() as u32).to_be_bytes());
        for n in &syncs {
            p.extend_from_slice(&n.to_be_bytes());
        }
        full_box(b"stss", 0, 0, &p)
    };

    // stsc: one sample per chunk, so one entry covers the whole track.
    let mut stsc = Vec::new();
    stsc.extend_from_slice(&1u32.to_be_bytes()); // entry count
    stsc.extend_from_slice(&1u32.to_be_bytes()); // first chunk
    stsc.extend_from_slice(&1u32.to_be_bytes()); // samples per chunk
    stsc.extend_from_slice(&1u32.to_be_bytes()); // sample description index

    // stsz: explicit per-sample sizes (default_size 0).
    let mut stsz = Vec::new();
    stsz.extend_from_slice(&0u32.to_be_bytes()); // default sample size
    stsz.extend_from_slice(&(idx.len() as u32).to_be_bytes());
    for &i in &idx {
        stsz.extend_from_slice(&(samples[i].bytes.len() as u32).to_be_bytes());
    }

    // stco / co64: the chunk (= sample) file offsets, 64-bit only if one does
    // not fit in 32 or the caller pinned the width.
    let needs_co64 = force_co64 || idx.iter().any(|&i| offsets[i] > u64::from(u32::MAX));
    let mut chunks = Vec::new();
    chunks.extend_from_slice(&(idx.len() as u32).to_be_bytes());
    for &i in &idx {
        if needs_co64 {
            chunks.extend_from_slice(&offsets[i].to_be_bytes());
        } else {
            chunks.extend_from_slice(&(offsets[i] as u32).to_be_bytes());
        }
    }
    let chunk_box = full_box(if needs_co64 { b"co64" } else { b"stco" }, 0, 0, &chunks);

    let media_time = u64::from(track_pre_skip(init));
    let timescale = init.timescale();
    let presentation = media_duration.saturating_sub(media_time);
    TrackTables {
        tables: [
            full_box(b"stts", 0, 0, &stts),
            ctts,
            stss,
            full_box(b"stsc", 0, 0, &stsc),
            full_box(b"stsz", 0, 0, &stsz),
            chunk_box,
        ]
        .concat(),
        media_duration,
        track_duration: presentation * u64::from(MOVIE_TIMESCALE) / u64::from(timescale.max(1)),
    }
}

/// The media-specific boxes of a track (the part that differs between video and
/// audio); the surrounding `trak`/`mdia`/`minf` scaffolding is shared.
struct TrakMedia {
    handler: &'static [u8; 4],
    /// `vmhd` (video) or `smhd` (audio).
    media_header: Vec<u8>,
    sample_entry: Vec<u8>,
    timescale: u32,
    dims: (u32, u32),
    is_video: bool,
}

fn trak_media(init: &TrackInit) -> TrakMedia {
    match init {
        TrackInit::Video {
            codec,
            width,
            height,
            param_sets,
        } => {
            let sample_entry = match codec {
                VideoCodec::Vp8 | VideoCodec::Vp9 => vp_sample_entry(*codec, *width, *height),
                VideoCodec::Av1 => {
                    // The captured init is the sequence-header OBU (see
                    // `capture_init`); its parse succeeded at capture.
                    let obu: &[u8] = param_sets.first().map(|v| v.as_slice()).unwrap_or(&[]);
                    let record = crate::av1parse::seq_header_obu(obu)
                        .map(|(seq, _)| av1c_record(&seq, obu))
                        .unwrap_or_default();
                    av1_sample_entry(*width, *height, &record)
                }
                _ => {
                    let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
                    visual_sample_entry(*codec, *width, *height, &refs)
                }
            };
            TrakMedia {
                handler: b"vide",
                media_header: full_box(b"vmhd", 0, 1, &[0u8; 8]),
                sample_entry,
                timescale: VIDEO_TIMESCALE,
                dims: (*width, *height),
                is_video: true,
            }
        }
        TrackInit::ClosedCaption { format } => TrakMedia {
            handler: b"clcp",
            media_header: caption_media_header(),
            sample_entry: cc_sample_entry(*format),
            timescale: VIDEO_TIMESCALE,
            dims: (0, 0),
            is_video: false,
        },
        // `sbtl` + `nmhd` is what ffmpeg writes for a `tx3g` track, and one of the
        // handlers the g2g demuxer reads a text sample entry under.
        TrackInit::Text => TrakMedia {
            handler: b"sbtl",
            media_header: full_box(b"nmhd", 0, 0, &[]),
            sample_entry: tx3g_sample_entry(),
            timescale: TEXT_TIMESCALE,
            dims: (0, 0),
            is_video: false,
        },
        TrackInit::Audio {
            format,
            channels,
            rate,
            config,
        } => {
            let sample_entry = match format {
                AudioFormat::Opus => {
                    audio_sample_entry(b"Opus", *channels, *rate, &dops(*channels, *rate, config))
                }
                _ => audio_sample_entry(b"mp4a", *channels, *rate, &esds(config)),
            };
            TrakMedia {
                handler: b"soun",
                media_header: full_box(b"smhd", 0, 0, &[0u8; 4]),
                sample_entry,
                timescale: *rate,
                dims: (0, 0),
                is_video: false,
            }
        }
    }
}

/// One raw-caption sample: the `cc_data` triples a
/// [`Caps::ClosedCaption`](g2g_core::Caps::ClosedCaption) frame carries, re-framed
/// into the atoms
/// the QuickTime caption sample entry defines. `c608` splits the triples by
/// line-21 field into a `cdat` (field 1) and a `cdt2` (field 2) atom, each holding
/// raw byte pairs; `c708` wraps them in a SMPTE ST 334-2 caption distribution
/// packet inside a `ccdp` atom. `seq` is the CDP sequence counter (608 ignores it).
fn cc_sample(cc_data: &[u8], format: ClosedCaptionFormat, seq: u16) -> Vec<u8> {
    let triples = parse_cc_data(cc_data);
    match format {
        ClosedCaptionFormat::Cea708 => {
            let dtvcc: Vec<CcTriple> = triples.iter().copied().filter(|t| t.cc_type >= 2).collect();
            if dtvcc.is_empty() {
                return Vec::new();
            }
            mp4_box(b"ccdp", &build_cdp(&dtvcc, CDP_FRAME_RATE_2997, seq))
        }
        // 608 (and any carriage this writer does not know, which the caps solver
        // never negotiates onto the pad).
        _ => {
            let mut out = Vec::new();
            for (cc_type, fourcc) in [(0u8, b"cdat"), (1u8, b"cdt2")] {
                let mut pairs = Vec::new();
                for t in triples.iter().filter(|t| t.cc_type == cc_type) {
                    pairs.push(t.b0);
                    pairs.push(t.b1);
                }
                if !pairs.is_empty() {
                    out.extend_from_slice(&mp4_box(fourcc, &pairs));
                }
            }
            out
        }
    }
}

/// A closed-caption sample entry (`c608` / `c708`): the plain `SampleEntry` fields
/// only (reserved + data reference index), no codec-configuration child.
fn cc_sample_entry(format: ClosedCaptionFormat) -> Vec<u8> {
    let fourcc: &[u8; 4] = match format {
        ClosedCaptionFormat::Cea708 => b"c708",
        _ => b"c608",
    };
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    mp4_box(fourcc, &p)
}

/// One 3GPP timed-text (`tx3g`) sample: a 2-byte big-endian text length then that
/// many UTF-8 bytes (TS 26.245), the inverse of the demuxer's de-framing. No
/// style / modifier boxes follow, so the cue is drawn with the sample entry's
/// default style. An empty text is the "no subtitle on screen" sample that spans
/// the gap between cues. A cue past what the length field can count is truncated
/// (a reader would mis-frame the sample otherwise).
fn tx3g_sample(text: &[u8]) -> Vec<u8> {
    let len = text.len().min(u16::MAX as usize);
    let mut out = (len as u16).to_be_bytes().to_vec();
    out.extend_from_slice(&text[..len]);
    out
}

/// The 3GPP timed-text sample entry (`tx3g`, TS 26.245): the plain `SampleEntry`
/// header, then the display flags, justification, background colour, the default
/// text box and style record, and a font table naming the style's font. The
/// values are the neutral defaults ffmpeg writes (centered at the bottom, opaque
/// white, 18 point); the text box stays empty because a text pad carries no video
/// geometry to size it against, which leaves the placement to the player.
fn tx3g_sample_entry() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&0u32.to_be_bytes()); // displayFlags
    p.push(0x01); // horizontal justification: centered
    p.push(0xFF); // vertical justification: bottom
    p.extend_from_slice(&[0, 0, 0, 0]); // background colour rgba
    p.extend_from_slice(&[0u8; 8]); // BoxRecord: top / left / bottom / right
    p.extend_from_slice(&0u16.to_be_bytes()); // style start char
    p.extend_from_slice(&0u16.to_be_bytes()); // style end char
    p.extend_from_slice(&1u16.to_be_bytes()); // font ID
    p.push(0); // face style flags
    p.push(18); // font size
    p.extend_from_slice(&[0xFF; 4]); // text colour rgba
    let mut ftab = 1u16.to_be_bytes().to_vec(); // font entry count
    ftab.extend_from_slice(&1u16.to_be_bytes()); // font ID
    ftab.push(5);
    ftab.extend_from_slice(b"Serif");
    p.extend_from_slice(&mp4_box(b"ftab", &ftab));
    mp4_box(b"tx3g", &p)
}

/// The `gmhd` base-media information a QuickTime caption track carries in place of
/// a `vmhd` / `smhd`: a `gmin` with the caption graphics mode and opcolor.
fn caption_media_header() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&0x0040u16.to_be_bytes()); // graphics mode
    p.extend_from_slice(&[0x80, 0x00, 0x80, 0x00, 0x80, 0x00]); // opcolor
    p.extend_from_slice(&0u16.to_be_bytes()); // balance
    p.extend_from_slice(&0u16.to_be_bytes()); // reserved
    mp4_box(b"gmhd", &full_box(b"gmin", 0, 0, &p))
}

/// The VP8/VP9 `VisualSampleEntry` (`vp08` / `vp09`) with its `vpcC`
/// VPCodecConfigurationBox (the VP-in-ISOBMFF binding). Geometry comes from the
/// caps; the codec config uses the 8-bit 4:2:0 unspecified-colour defaults (no
/// bitstream parsing), which a player overrides from the frames it decodes.
fn vp_sample_entry(codec: VideoCodec, width: u32, height: u32) -> Vec<u8> {
    let fourcc: &[u8; 4] = match codec {
        VideoCodec::Vp8 => b"vp08",
        _ => b"vp09",
    };
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 16]); // pre_defined / reserved
    p.extend_from_slice(&(width as u16).to_be_bytes());
    p.extend_from_slice(&(height as u16).to_be_bytes());
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    p.extend_from_slice(&[0u8; 4]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // frame count
    p.extend_from_slice(&[0u8; 32]); // compressor name
    p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
    p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    p.extend_from_slice(&vpcc());
    mp4_box(fourcc, &p)
}

/// The AV1 `VisualSampleEntry` (`av01`) with its `av1C`
/// AV1CodecConfigurationBox (the AV1-in-ISOBMFF binding). Geometry comes from
/// the caps; the record is built from the stream's own sequence header.
fn av1_sample_entry(width: u32, height: u32, av1c: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 16]); // pre_defined / reserved
    p.extend_from_slice(&(width as u16).to_be_bytes());
    p.extend_from_slice(&(height as u16).to_be_bytes());
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    p.extend_from_slice(&[0u8; 4]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // frame count
    p.extend_from_slice(&[0u8; 32]); // compressor name
    p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
    p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    p.extend_from_slice(&mp4_box(b"av1C", av1c));
    mp4_box(b"av01", &p)
}

/// The `vpcC` VPCodecConfigurationBox (fullbox v1): profile 0, level unset, 8-bit
/// 4:2:0 (colocated), unspecified colour, no codec initialization data, the
/// generic defaults for an unparsed VP8/VP9 stream.
fn vpcc() -> Vec<u8> {
    // profile 0, level unset, then a packed byte:
    // bitDepth(4)=8 | chromaSubsampling(3)=1 (4:2:0 colocated) | videoFullRangeFlag(1)=0,
    // colour_primaries / transfer / matrix unspecified (2), then a 2-byte
    // codec_initialization_data_size of 0 (VP8/VP9 carry no init data).
    let record = [0u8, 0, (8 << 4) | (1 << 1), 2, 2, 2, 0, 0];
    full_box(b"vpcC", 1, 0, &record)
}

/// An `AudioSampleEntry` box (`mp4a` / `Opus`): the shared sample-entry header
/// then the codec-specific config box (`esds` / `dOps`).
fn audio_sample_entry(fourcc: &[u8; 4], channels: u8, rate: u32, config: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 8]); // reserved
    p.extend_from_slice(&(channels as u16).to_be_bytes());
    p.extend_from_slice(&16u16.to_be_bytes()); // sample size
    p.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    p.extend_from_slice(&0u16.to_be_bytes()); // reserved
    p.extend_from_slice(&(rate << 16).to_be_bytes()); // 16.16 sample rate
    p.extend_from_slice(config);
    mp4_box(fourcc, &p)
}

/// The `dOps` OpusSpecificBox (RFC 8316): the Opus init data in an MP4 audio
/// sample entry. Fields are big-endian (unlike the little-endian Ogg/WebM
/// `OpusHead`). Built from the stream's own `OpusHead` when one arrived in band,
/// so a remux keeps the source's pre-skip, output gain and channel mapping;
/// otherwise the caps plus libopus' encoder lookahead, mapping family 0.
fn dops(channels: u8, rate: u32, head: &[u8]) -> Vec<u8> {
    let body = dops_from_opus_head(head).unwrap_or_else(|| {
        let mut b = Vec::new();
        b.push(0); // Version
        b.push(channels.max(1)); // OutputChannelCount
        b.extend_from_slice(&OPUS_ENCODER_PRE_SKIP.to_be_bytes()); // PreSkip
        b.extend_from_slice(&rate.to_be_bytes()); // InputSampleRate
        b.extend_from_slice(&0i16.to_be_bytes()); // OutputGain
        b.push(0); // ChannelMappingFamily
        b
    });
    mp4_box(b"dOps", &body)
}

/// The pre-skip a track's `dOps` declares, which its edit list must skip. `0`
/// for a non-Opus track.
fn track_pre_skip(init: &TrackInit) -> u32 {
    match init {
        TrackInit::Audio {
            format: AudioFormat::Opus,
            config,
            ..
        } => parse_opus_head(config)
            .map(|(_, pre_skip)| u32::from(pre_skip))
            .unwrap_or(u32::from(OPUS_ENCODER_PRE_SKIP)),
        _ => 0,
    }
}

/// The `edts`/`elst` that trims a track's codec delay off the presentation
/// timeline (Opus-in-ISOBMFF): `media_time` is the pre-skip in media timescale
/// units, so playback starts at the first real sample instead of the decoder's
/// pre-roll. `segment_duration` (movie timescale) is the presentation length,
/// or `0` ("to the end of the media") in the fragmented layout, whose `moov` is
/// written before the total is known. Empty for a track with no delay to trim.
fn edts(init: &TrackInit, segment_duration: u64) -> Vec<u8> {
    let media_time = track_pre_skip(init);
    if media_time == 0 {
        return Vec::new();
    }
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_be_bytes()); // entry count
    p.extend_from_slice(&(segment_duration as u32).to_be_bytes());
    p.extend_from_slice(&media_time.to_be_bytes());
    p.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // media rate 1.0
    mp4_box(b"edts", &full_box(b"elst", 0, 0, &p))
}

/// One `trak` for a track (`track_ID` 1-based). `tables` carries the
/// progressive layout's sample tables and durations; `None` writes the
/// fragmented form (empty tables, zero durations, the fragments carry timing).
fn trak(
    track_id: u32,
    init: &TrackInit,
    tables: Option<&TrackTables>,
    tags: Option<&TagList>,
) -> Vec<u8> {
    let TrakMedia {
        handler,
        media_header: header,
        sample_entry,
        timescale,
        dims,
        is_video,
    } = trak_media(init);
    let tkhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]); // times
        p.extend_from_slice(&track_id.to_be_bytes());
        p.extend_from_slice(&[0u8; 4]); // reserved
        let track_duration = tables.map(|t| t.track_duration).unwrap_or(0);
        p.extend_from_slice(&(track_duration as u32).to_be_bytes()); // movie timescale
        p.extend_from_slice(&[0u8; 8]); // reserved
        p.extend_from_slice(&0u16.to_be_bytes()); // layer
        p.extend_from_slice(&0u16.to_be_bytes()); // alternate group
                                                  // audio tracks carry volume 1.0, video tracks 0.
        p.extend_from_slice(&(if is_video { 0u16 } else { 0x0100 }).to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&(dims.0 << 16).to_be_bytes()); // 16.16 width
        p.extend_from_slice(&(dims.1 << 16).to_be_bytes()); // 16.16 height
        full_box(b"tkhd", 0, 3, &p) // enabled | in_movie
    };

    let mdhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&timescale.to_be_bytes());
        let media_duration = tables.map(|t| t.media_duration).unwrap_or(0);
        p.extend_from_slice(&(media_duration as u32).to_be_bytes());
        p.extend_from_slice(&0x55C4u16.to_be_bytes()); // language "und"
        p.extend_from_slice(&[0u8; 2]);
        full_box(b"mdhd", 0, 0, &p)
    };
    let hdlr = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 4]);
        p.extend_from_slice(handler);
        p.extend_from_slice(&[0u8; 12]);
        p.extend_from_slice(b"g2g\0");
        full_box(b"hdlr", 0, 0, &p)
    };
    let stbl = {
        let stsd = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes());
            p.extend_from_slice(&sample_entry);
            full_box(b"stsd", 0, 0, &p)
        };
        // A fragmented file's tables are empty (the `trun`s carry the samples);
        // a progressive one indexes every sample.
        let sample_tables = match tables {
            Some(t) => t.tables.clone(),
            None => {
                let empty4 = 0u32.to_be_bytes();
                [
                    full_box(b"stts", 0, 0, &empty4),
                    full_box(b"stsc", 0, 0, &empty4),
                    full_box(b"stsz", 0, 0, &[0u8; 8]),
                    full_box(b"stco", 0, 0, &empty4),
                ]
                .concat()
            }
        };
        mp4_box(b"stbl", &[stsd, sample_tables].concat())
    };
    let dinf = {
        let url = full_box(b"url ", 0, 1, &[]);
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&url);
        let dref = full_box(b"dref", 0, 0, &p);
        mp4_box(b"dinf", &dref)
    };
    let minf = mp4_box(b"minf", &[header, dinf, stbl].concat());
    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
    let segment_duration = tables.map(|t| t.track_duration).unwrap_or(0);
    // This track's own metadata, which a reader reports on the elementary stream
    // rather than the file (M838).
    let udta = tags
        .and_then(|t| udta_with_tags(t, &[]))
        .unwrap_or_default();
    mp4_box(
        b"trak",
        &[tkhd, edts(init, segment_duration), mdia, udta].concat(),
    )
}

/// One `moof`+`mdat` fragment holding `samples` for `track_id`, with a
/// multi-sample `trun` (a single sample in the default per-AU mode). `tfdt` is the
/// track's decode time at the first sample.
fn av_fragment(
    track_id: u32,
    sequence: u64,
    base_decode_time: u64,
    cmaf: bool,
    samples: &[PendingSample],
) -> Vec<u8> {
    let build_moof = |data_offset: u32| -> Vec<u8> {
        let mfhd = full_box(b"mfhd", 0, 0, &(sequence as u32).to_be_bytes());
        let tfhd = tfhd(track_id, cmaf);
        let tfdt = full_box(b"tfdt", 1, 0, &base_decode_time.to_be_bytes());
        let trun = {
            let mut p = Vec::new();
            p.extend_from_slice(&(samples.len() as u32).to_be_bytes()); // sample count
            p.extend_from_slice(&data_offset.to_be_bytes());
            for s in samples {
                let flags: u32 = if s.is_sync { 0x0200_0000 } else { 0x0101_0000 };
                p.extend_from_slice(&s.duration.to_be_bytes());
                p.extend_from_slice(&(s.bytes.len() as u32).to_be_bytes());
                p.extend_from_slice(&flags.to_be_bytes());
            }
            full_box(b"trun", 0, 0x000701, &p) // data-offset | duration | size | flags (per sample)
        };
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        mp4_box(b"moof", &[mfhd, traf].concat())
    };
    let moof_len = build_moof(0).len() as u32;
    let moof = build_moof(moof_len + 8);
    let mut mdat_payload = Vec::new();
    for s in samples {
        mdat_payload.extend_from_slice(&s.bytes);
    }
    let mdat = mp4_box(b"mdat", &mdat_payload);
    [moof, mdat].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    fn h264_caps(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
        }
    }

    /// Count `moof` fragment boxes and sum every `trun`'s sample count.
    fn moof_and_sample_count(bytes: &[u8]) -> (usize, u64) {
        let moofs = bytes.windows(4).filter(|w| *w == b"moof").count();
        let mut samples = 0u64;
        for (i, w) in bytes.windows(4).enumerate() {
            if w == b"trun" {
                let off = i + 8; // [..'trun'][version+flags:4][sample_count:4]
                if off + 4 <= bytes.len() {
                    samples += u32::from_be_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]) as u64;
                }
            }
        }
        (moofs, samples)
    }

    /// The `fragment-duration` knob is honored on the fan-in muxer (the `name=m`
    /// shape), the same as the single-track `Mp4Mux`: access units batch into
    /// keyframe-aligned multi-sample fragments, versus one fragment per AU by
    /// default. Set via `set_property`, the path `parse_launch` uses.
    #[tokio::test]
    async fn fragment_duration_property_batches_on_the_fan_in_muxer() {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        let key = annexb(&[&sps, &pps, &idr]);
        let inter = || annexb(&[&[0x41u8, 0x9a, 0x00]]);
        let aus = [key.clone(), inter(), inter(), inter(), inter(), key.clone()];

        // Batched: a 10 ms target (frames are ~33 ms) closes the fragment at the
        // next IDR, so AU0..AU4 form one fragment and AU5 the next (flushed at EOS).
        let mut mux = Mp4MuxN::new(1);
        mux.set_property("fragment-duration", PropValue::Uint(10))
            .unwrap();
        assert_eq!(
            mux.get_property("fragment-duration"),
            Some(PropValue::Uint(10))
        );
        mux.configure_pipeline(0, &h264_caps(320, 240)).unwrap();
        let mut sink = CaptureSink::default();
        for (i, au) in aus.iter().enumerate() {
            mux.process(0, frame(au.clone(), i as u64 * 33_333_333), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        let (moofs, samples) = moof_and_sample_count(&sink.bytes);
        assert_eq!(
            moofs, 2,
            "six AUs batch into two keyframe-aligned fragments"
        );
        assert_eq!(samples, 6, "every access unit is preserved as a sample");

        // Default (per-AU): one fragment per access unit, byte-for-byte as before.
        let mut mux0 = Mp4MuxN::new(1);
        mux0.configure_pipeline(0, &h264_caps(320, 240)).unwrap();
        let mut sink0 = CaptureSink::default();
        for (i, au) in aus.iter().enumerate() {
            mux0.process(0, frame(au.clone(), i as u64 * 33_333_333), &mut sink0)
                .await
                .unwrap();
        }
        mux0.process(0, PipelinePacket::Eos, &mut sink0)
            .await
            .unwrap();
        let (moofs0, samples0) = moof_and_sample_count(&sink0.bytes);
        assert_eq!(moofs0, 6, "per-AU mode emits one fragment per access unit");
        assert_eq!(samples0, 6);
    }

    #[test]
    fn asc_from_adts_recovers_lc_params() {
        // 48 kHz (index 3), stereo (2), LC: byte2 = (1<<6)|(3<<2) = 0x4C, byte3 high
        // 2 bits = channel 2 -> 0x80.
        let adts = [0xFF, 0xF1, 0x4C, 0x80, 0x00, 0x00, 0x00, 0xDE, 0xAD];
        let asc = asc_from_adts(&adts).expect("valid adts");
        // ASC: AOT=2(00010), srIndex=3(0011), chan=2(0010), pad.
        // byte0 = (2<<3)|(3>>1) = 0x10|0x01 = 0x11; byte1 = ((3&1)<<7)|(2<<3) = 0x80|0x10 = 0x90.
        assert_eq!(asc, [0x11, 0x90]);
    }

    #[test]
    fn strip_adts_removes_7_byte_header() {
        let adts = [0xFF, 0xF1, 0x4C, 0x80, 0x00, 0x00, 0x00, 0xAA, 0xBB];
        assert_eq!(strip_adts(&adts), &[0xAA, 0xBB]);
        // a non-ADTS payload is returned unchanged
        assert_eq!(strip_adts(&[1, 2, 3]), &[1, 2, 3]);
    }

    #[test]
    fn moov_has_two_traks_and_two_trex() {
        let tracks = [
            Some(TrackInit::Video {
                codec: VideoCodec::H264,
                width: 320,
                height: 240,
                param_sets: alloc::vec![
                    alloc::vec![0x67, 0x42, 0x00, 0x1e],
                    alloc::vec![0x68, 0xce]
                ],
            }),
            Some(TrackInit::Audio {
                format: AudioFormat::Aac,
                channels: 2,
                rate: 48000,
                config: alloc::vec![0x11, 0x90],
            }),
        ];
        let moov = av_moov(&tracks, None, &MoovMetadata::default());
        let count = |needle: &[u8]| moov.windows(4).filter(|w| *w == needle).count();
        assert_eq!(count(b"trak"), 2, "one trak per track");
        assert_eq!(count(b"trex"), 2, "one trex per track");
        assert_eq!(count(b"avcC"), 1, "video sample entry");
        assert_eq!(count(b"esds"), 1, "audio sample entry");
        assert_eq!(count(b"soun"), 1);
        assert_eq!(count(b"vide"), 1);
    }

    #[test]
    fn empty_leading_pad_keeps_track_id_aligned() {
        // Pad slot 0 produced no AU; the audio pad sits in slot 1, so its trak
        // and trex must carry track_ID 2 to match the fragments emit_au writes.
        let tracks = [
            None,
            Some(TrackInit::Audio {
                format: AudioFormat::Aac,
                channels: 2,
                rate: 48000,
                config: alloc::vec![0x11, 0x90],
            }),
        ];
        let moov = av_moov(&tracks, None, &MoovMetadata::default());
        let count = |needle: &[u8]| moov.windows(4).filter(|w| *w == needle).count();
        assert_eq!(count(b"trak"), 1, "only the non-empty pad gets a trak");
        assert_eq!(count(b"trex"), 1);
        let trex = moov.windows(4).position(|w| w == b"trex").unwrap();
        let track_id = u32::from_be_bytes(moov[trex + 8..trex + 12].try_into().unwrap());
        assert_eq!(
            track_id, 2,
            "slot-1 pad keeps track_ID 2 despite the empty slot 0"
        );
    }

    #[test]
    fn opus_track_writes_an_opus_sample_entry_with_dops() {
        let tracks = [Some(TrackInit::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            rate: 48000,
            config: Vec::new(),
        })];
        let moov = av_moov(&tracks, None, &MoovMetadata::default());
        let count = |needle: &[u8]| moov.windows(needle.len()).filter(|w| *w == needle).count();
        assert_eq!(count(b"Opus"), 1, "Opus sample entry");
        assert_eq!(count(b"dOps"), 1, "OpusSpecificBox");
        assert_eq!(count(b"esds"), 0, "no AAC descriptor for an Opus track");
        assert_eq!(count(b"soun"), 1, "sound handler");
        // With no in-band header the dOps declares libopus' lookahead, and the
        // edit list skips exactly that much media.
        let dops = moov.windows(4).position(|w| w == b"dOps").unwrap();
        assert_eq!(
            u16::from_be_bytes([moov[dops + 6], moov[dops + 7]]),
            OPUS_ENCODER_PRE_SKIP,
            "synthesized PreSkip"
        );
        assert_eq!(count(b"elst"), 1, "edit list on the Opus track");
        let elst = moov.windows(4).position(|w| w == b"elst").unwrap();
        assert_eq!(
            u32::from_be_bytes(moov[elst + 16..elst + 20].try_into().unwrap()),
            u32::from(OPUS_ENCODER_PRE_SKIP),
            "elst media_time is the pre-skip"
        );
    }

    /// An in-band `OpusHead` is the authority: its pre-skip reaches the `dOps`
    /// and the edit list instead of the synthesized fallback.
    #[test]
    fn an_in_band_opus_head_sets_the_dops_pre_skip() {
        let mut head = Vec::from(*b"OpusHead");
        head.extend_from_slice(&[1, 2]); // version, channels
        head.extend_from_slice(&666u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&[0, 0, 0]); // output gain, mapping family
        let tracks = [Some(TrackInit::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            rate: 48000,
            config: head,
        })];
        let moov = av_moov(&tracks, None, &MoovMetadata::default());
        let dops = moov.windows(4).position(|w| w == b"dOps").unwrap();
        assert_eq!(
            u16::from_be_bytes([moov[dops + 6], moov[dops + 7]]),
            666,
            "the header's pre-skip, not the fallback"
        );
        let elst = moov.windows(4).position(|w| w == b"elst").unwrap();
        assert_eq!(
            u32::from_be_bytes(moov[elst + 16..elst + 20].try_into().unwrap()),
            666,
            "the elst follows the header"
        );
    }

    #[test]
    fn vp9_track_writes_a_vp09_sample_entry_with_vpcc() {
        let tracks = [Some(TrackInit::Video {
            codec: VideoCodec::Vp9,
            width: 320,
            height: 240,
            param_sets: Vec::new(),
        })];
        let moov = av_moov(&tracks, None, &MoovMetadata::default());
        let count = |needle: &[u8]| moov.windows(needle.len()).filter(|w| *w == needle).count();
        assert_eq!(count(b"vp09"), 1, "VP9 sample entry");
        assert_eq!(count(b"vpcC"), 1, "VPCodecConfigurationBox");
        assert_eq!(count(b"avcC"), 0, "no avcC for a VP9 track");
        assert_eq!(count(b"vide"), 1, "video handler");
    }

    /// The top-level 4ccs of `file`, in order.
    fn top_level_boxes(file: &[u8]) -> Vec<[u8; 4]> {
        let mut out = Vec::new();
        let mut at = 0usize;
        while at + 8 <= file.len() {
            let size = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
            if size < 8 || at + size > file.len() {
                break;
            }
            out.push(file[at + 4..at + 8].try_into().unwrap());
            at += size;
        }
        out
    }

    /// The payload of the first box named `fourcc`, or `None`.
    fn box_payload<'a>(file: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
        let at = file.windows(4).position(|w| w == fourcc)?;
        let size = u32::from_be_bytes(file[at - 4..at].try_into().unwrap()) as usize;
        file.get(at + 4..at - 4 + size)
    }

    /// Six H.264 access units (two IDRs, four inter frames) through the muxer in
    /// the given layout.
    async fn mux_six_aus(fragmented: bool) -> Vec<u8> {
        mux_six_aus_faststart(fragmented, false).await
    }

    async fn mux_six_aus_faststart(fragmented: bool, faststart: bool) -> Vec<u8> {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        let key = annexb(&[&sps, &pps, &idr]);
        let inter = || annexb(&[&[0x41u8, 0x9a, 0x00]]);
        let aus = [key.clone(), inter(), inter(), inter(), inter(), key];

        let mut mux = Mp4MuxN::new(1)
            .with_fragmented(fragmented)
            .with_faststart(faststart);
        mux.configure_pipeline(0, &h264_caps(320, 240)).unwrap();
        let mut sink = CaptureSink::default();
        for (i, au) in aus.iter().enumerate() {
            mux.process(0, frame(au.clone(), i as u64 * 33_333_333), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        sink.bytes
    }

    /// The progressive layout is `ftyp` + one `mdat` + a `moov` with real sample
    /// tables and no `mvex`; the fragmented default is unchanged.
    #[tokio::test]
    async fn progressive_writes_ftyp_mdat_moov_with_real_sample_tables() {
        let file = mux_six_aus(false).await;
        let boxes = top_level_boxes(&file);
        assert_eq!(
            boxes,
            alloc::vec![*b"ftyp", *b"mdat", *b"moov"],
            "one mdat between the brands and the index"
        );
        let count = |needle: &[u8]| file.windows(needle.len()).filter(|w| *w == needle).count();
        assert_eq!(count(b"moof"), 0, "no fragments");
        assert_eq!(count(b"mvex"), 0, "and nothing announcing any");
        for table in [b"stts", b"stsc", b"stsz", b"stco", b"stss"] {
            assert_eq!(
                count(table),
                1,
                "one {} table",
                core::str::from_utf8(table).unwrap()
            );
        }
        assert_eq!(count(b"ctts"), 0, "pts == dts, so no composition offsets");

        // stsz: six samples, none empty.
        let stsz = box_payload(&file, b"stsz").expect("stsz");
        assert_eq!(
            u32::from_be_bytes(stsz[8..12].try_into().unwrap()),
            6,
            "every access unit is indexed"
        );
        // stss: the two IDRs are the sync samples (1-based).
        let stss = box_payload(&file, b"stss").expect("stss");
        assert_eq!(u32::from_be_bytes(stss[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_be_bytes(stss[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(stss[12..16].try_into().unwrap()), 6);
        // The movie lasts six 33.3 ms frames: 2999 ticks each at 90 kHz, so
        // 199 ms in the 1 kHz movie timescale.
        let mvhd = box_payload(&file, b"mvhd").expect("mvhd");
        assert_eq!(u32::from_be_bytes(mvhd[16..20].try_into().unwrap()), 199);

        // Sample bytes really live where stco says they do: the first chunk
        // offset lands on the first sample's AVCC length prefix.
        let stco = box_payload(&file, b"stco").expect("stco");
        let first = u32::from_be_bytes(stco[8..12].try_into().unwrap()) as usize;
        let first_len = u32::from_be_bytes(stsz[12..16].try_into().unwrap()) as usize;
        assert_eq!(
            u32::from_be_bytes(file[first..first + 4].try_into().unwrap()) as usize,
            5,
            "the first sample starts with its SPS's 4-byte AVCC length"
        );
        assert!(first + first_len <= file.len(), "the sample is in the file");

        // The default is untouched.
        let frag = mux_six_aus(true).await;
        assert!(
            frag.windows(4).any(|w| w == b"moof"),
            "the fragmented default still writes fragments"
        );
        assert_eq!(
            top_level_boxes(&frag)[..2],
            [*b"ftyp", *b"moov"],
            "and still leads with the init segment"
        );
    }

    /// M824 faststart: the same progressive file with its `moov` ahead of the
    /// `mdat`, every chunk offset shifted by the `moov`'s own size so the sample
    /// bytes are still where `stco` points.
    #[tokio::test]
    async fn faststart_moves_the_moov_ahead_of_the_mdat() {
        let file = mux_six_aus_faststart(false, true).await;
        assert_eq!(
            top_level_boxes(&file),
            alloc::vec![*b"ftyp", *b"moov", *b"mdat"],
            "the index precedes the media"
        );

        let plain = mux_six_aus_faststart(false, false).await;
        assert_eq!(file.len(), plain.len(), "the same boxes, reordered");

        // Every sample really lives at its stco offset: the first one starts
        // with its SPS's 4-byte AVCC length, and the last ends the file.
        let stco = box_payload(&file, b"stco").expect("stco");
        let stsz = box_payload(&file, b"stsz").expect("stsz");
        let count = u32::from_be_bytes(stsz[8..12].try_into().unwrap()) as usize;
        assert_eq!(count, 6);
        let mut end = 0usize;
        for i in 0..count {
            let off = u32::from_be_bytes(stco[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
            let len = u32::from_be_bytes(stsz[12 + i * 4..16 + i * 4].try_into().unwrap()) as usize;
            if i > 0 {
                assert_eq!(off, end, "samples are packed in mdat order");
            }
            end = off + len;
        }
        assert_eq!(end, file.len(), "the last sample ends the file");

        // The shift is exactly the moov's size: the moov-at-end file's first
        // offset plus the moov length is the faststart file's first offset.
        let first = u32::from_be_bytes(stco[8..12].try_into().unwrap());
        let plain_stco = box_payload(&plain, b"stco").expect("plain stco");
        let plain_first = u32::from_be_bytes(plain_stco[8..12].try_into().unwrap());
        let moov_len = box_payload(&file, b"moov").expect("moov").len() as u32 + 8;
        assert_eq!(first, plain_first + moov_len);
        assert_eq!(
            u32::from_be_bytes(file[first as usize..first as usize + 4].try_into().unwrap()),
            5,
            "the first sample starts with its SPS's 4-byte AVCC length"
        );

        // A fragmented file's moov already leads, so faststart changes nothing.
        assert_eq!(
            mux_six_aus_faststart(true, true).await,
            mux_six_aus_faststart(true, false).await,
            "the fragmented layout is untouched"
        );
    }

    /// A stream whose frames decode before they present gets a `ctts`, one
    /// run-length entry for the constant offset.
    #[tokio::test]
    async fn progressive_writes_ctts_when_decode_leads_presentation() {
        let idr = annexb(&[
            &[0x67u8, 0x42, 0x00, 0x1e, 0x88],
            &[0x68u8, 0xce],
            &[0x65u8, 0x88],
        ]);
        let mut mux = Mp4MuxN::new(1).with_fragmented(false);
        mux.configure_pipeline(0, &h264_caps(320, 240)).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..3u64 {
            let pts = 100_000_000 + i * 33_333_333;
            let packet = PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(idr.clone().into_boxed_slice())),
                FrameTiming {
                    pts_ns: pts,
                    // Two frames of reorder delay.
                    dts_ns: pts - 66_666_666,
                    ..FrameTiming::default()
                },
                0,
            ));
            mux.process(0, packet, &mut sink).await.unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        let ctts = box_payload(&sink.bytes, b"ctts").expect("ctts written");
        assert_eq!(ctts.first(), Some(&0), "version 0, unsigned offsets");
        assert_eq!(
            u32::from_be_bytes(ctts[4..8].try_into().unwrap()),
            1,
            "one run covers the constant offset"
        );
        assert_eq!(u32::from_be_bytes(ctts[8..12].try_into().unwrap()), 3);
        assert_eq!(
            u32::from_be_bytes(ctts[12..16].try_into().unwrap()),
            ns_to_ts(66_666_666, VIDEO_TIMESCALE) as u32,
            "pts - dts in the video timescale"
        );
    }

    /// A demuxer negotiates compressed audio as `0/0` and refines at runtime;
    /// the muxer must take the refinement or write a zero `mdhd` timescale.
    #[tokio::test]
    async fn runtime_caps_refinement_sizes_the_audio_track() {
        let sentinel = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 0,
            sample_rate: 0,
        };
        let refined = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        };
        let mut mux = Mp4MuxN::new(1).with_fragmented(false);
        mux.configure_pipeline(0, &sentinel).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(0, PipelinePacket::CapsChanged(refined), &mut sink)
            .await
            .unwrap();
        for i in 0..3u64 {
            mux.process(
                0,
                frame(alloc::vec![0xFC, i as u8, 1, 2], i * 20_000_000),
                &mut sink,
            )
            .await
            .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();

        let mdhd = box_payload(&sink.bytes, b"mdhd").expect("mdhd");
        assert_eq!(
            u32::from_be_bytes(mdhd[12..16].try_into().unwrap()),
            48_000,
            "the refined sample rate is the media timescale"
        );
        assert!(
            u32::from_be_bytes(mdhd[16..20].try_into().unwrap()) > 0,
            "and the track has a duration"
        );
    }
}

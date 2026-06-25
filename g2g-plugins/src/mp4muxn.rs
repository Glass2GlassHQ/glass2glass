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
//! first IDR, an audio track's AudioSpecificConfig is synthesised from the first
//! ADTS header (the AAC bytes are written de-ADTS'd into the `mdat`). After the
//! init segment, one `moof`+`mdat` fragment per access unit, each `traf`
//! referencing its track with a per-track `tfdt` in that track's timescale.
//!
//! Reachable from the `gst-launch` fan-in syntax: registered as the `mp4mux`
//! muxer in `default_registry`, so >1 input link builds this element (a single
//! input builds the single-track [`crate::mp4mux::Mp4Mux`]), the way gst's
//! request sink pads do. Scope (v1): H.264/H.265 + AAC, sync-sample audio.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement, OutputSink,
    PipelinePacket, VideoCodec,
};

use crate::mp4box::{ftyp, full_box, mp4_box, MATRIX};
use crate::fmp4mux::{avcc_sample, is_keyframe_nal, parameter_sets, split_annexb, visual_sample_entry};
use crate::mp4audiosink::esds;

/// Video tracks use a 90 kHz media timescale; audio tracks use the sample rate.
const VIDEO_TIMESCALE: u32 = 90_000;
const DEFAULT_VIDEO_DURATION_NS: u64 = 33_333_333;

/// What an input pad carries, learned from its negotiated caps at configure.
#[derive(Debug, Clone, Copy)]
enum PadKind {
    Video(VideoCodec),
    Audio { format: AudioFormat, channels: u8, rate: u32 },
}

/// A track's `moov` init data, captured from its first access unit. `asc` is the
/// AAC AudioSpecificConfig (empty for Opus, whose `dOps` is built from the caps).
#[derive(Debug, Clone)]
enum TrackInit {
    Video { codec: VideoCodec, width: u32, height: u32, param_sets: Vec<Vec<u8>> },
    Audio { format: AudioFormat, channels: u8, rate: u32, asc: Vec<u8> },
}

impl TrackInit {
    fn timescale(&self) -> u32 {
        match self {
            TrackInit::Video { .. } => VIDEO_TIMESCALE,
            TrackInit::Audio { rate, .. } => *rate,
        }
    }
}

/// Muxes N elementary streams into one ISO-BMFF byte stream, PTS-ordered.
#[derive(Debug)]
pub struct Mp4MuxN {
    inputs: usize,
    /// Per-pad stream kind, learned at configure (the moov needs every track).
    kinds: Vec<Option<PadKind>>,
    /// Per-pad track init, captured from the first AU. Geometry comes from the
    /// caps; video parameter sets / audio ASC come in-band from the first AU.
    inits: Vec<Option<TrackInit>>,
    /// Per-pad caps geometry (video width/height), recorded at configure.
    dims: Vec<(u32, u32)>,
    agg: InputAggregator<Frame>,
    /// Per-track accumulated decode time in that track's timescale (`tfdt`).
    decode_time: Vec<u64>,
    /// Per-track previous PTS (ns), for the sample-duration delta.
    prev_pts_ns: Vec<Option<u64>>,
    header_written: bool,
    /// Global moof sequence number (1-based, increasing across the movie).
    sequence: u64,
    emitted: u64,
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
            header_written: false,
            sequence: 0,
            emitted: 0,
        }
    }

    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }

    fn pad_kind_for(caps: &Caps) -> Option<PadKind> {
        match caps {
            Caps::CompressedVideo { codec: c @ (VideoCodec::H264 | VideoCodec::H265), .. } => {
                Some(PadKind::Video(*c))
            }
            Caps::Audio {
                format: format @ (AudioFormat::Aac | AudioFormat::Opus),
                channels,
                sample_rate,
            } => Some(PadKind::Audio { format: *format, channels: *channels, rate: *sample_rate }),
            _ => None,
        }
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
                let nalus = split_annexb(au);
                // Parameter sets only ride the IDR; a leading P-frame has none, so
                // wait for the keyframe that carries them.
                if let Ok(param_sets) = parameter_sets(codec, &nalus) {
                    let owned: Vec<Vec<u8>> = param_sets.iter().map(|s| s.to_vec()).collect();
                    let (w, h) = self.dims[input];
                    self.inits[input] = Some(TrackInit::Video { codec, width: w, height: h, param_sets: owned });
                }
            }
            Some(PadKind::Audio { format, channels, rate }) => match format {
                // AAC's AudioSpecificConfig is synthesised from the first ADTS header.
                AudioFormat::Aac => {
                    if let Some(asc) = asc_from_adts(au) {
                        self.inits[input] =
                            Some(TrackInit::Audio { format, channels, rate, asc: asc.to_vec() });
                    }
                }
                // Opus needs no in-band init; its `dOps` comes from the caps.
                _ => {
                    self.inits[input] =
                        Some(TrackInit::Audio { format, channels, rate, asc: Vec::new() });
                }
            },
            None => {}
        }
    }

    /// The mdat sample bytes for a track: AVCC length-prefixed NALUs for video,
    /// the de-ADTS'd raw AAC for audio. Also returns whether it is a sync sample.
    fn sample_for(&self, input: usize, au: &[u8]) -> (Vec<u8>, bool) {
        match self.kinds[input] {
            Some(PadKind::Video(codec)) => {
                let nalus = split_annexb(au);
                let is_sync = nalus.iter().any(|n| is_keyframe_nal(codec, n));
                (avcc_sample(&nalus), is_sync)
            }
            // Audio access units are always sync samples. AAC strips its ADTS
            // header; Opus packets are stored raw.
            Some(PadKind::Audio { format: AudioFormat::Aac, .. }) => (strip_adts(au).to_vec(), true),
            _ => (au.to_vec(), true),
        }
    }

    /// Emit one access unit as a `moof`+`mdat` fragment for its track, prepending
    /// the `ftyp`+`moov` init segment on the first fragment.
    async fn emit_au(&mut self, input: usize, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let au = slice.as_slice();
        let pts_ns = frame.timing.pts_ns;
        let (sample, is_sync) = self.sample_for(input, au);

        let mut bytes = Vec::new();
        if !self.header_written {
            let tracks: Vec<TrackInit> = self.inits.iter().flatten().cloned().collect();
            bytes.extend_from_slice(&ftyp());
            bytes.extend_from_slice(&av_moov(&tracks));
            self.header_written = true;
        }

        let track = &self.inits[input];
        let timescale = track.as_ref().map(TrackInit::timescale).unwrap_or(VIDEO_TIMESCALE);
        let default_dur_ns = match self.kinds[input] {
            // Opus frames are 20 ms (960 samples @ 48 kHz); AAC frames 1024 samples.
            Some(PadKind::Audio { format: AudioFormat::Opus, rate, .. }) => {
                960 * 1_000_000_000 / rate.max(1) as u64
            }
            Some(PadKind::Audio { rate, .. }) => 1024 * 1_000_000_000 / rate.max(1) as u64,
            _ => DEFAULT_VIDEO_DURATION_NS,
        };
        let dur_ns = match self.prev_pts_ns[input] {
            Some(prev) if pts_ns > prev => pts_ns - prev,
            _ => default_dur_ns,
        };
        self.prev_pts_ns[input] = Some(pts_ns);
        let duration = ns_to_ts(dur_ns, timescale);

        // track_ID is the 1-based pad index.
        let track_id = (input + 1) as u32;
        self.sequence += 1;
        bytes.extend_from_slice(&av_fragment(track_id, self.sequence, self.decode_time[input], duration as u32, &sample, is_sync));
        self.decode_time[input] += duration;

        let out_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming { pts_ns, ..FrameTiming::default() },
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

    fn input_count(&self) -> usize {
        self.inputs
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
        Ok(CapsConstraint::Produces(CapsSet::one(Self::output_caps_value())))
    }

    fn configure_pipeline(&mut self, input: usize, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let kind = Self::pad_kind_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if let Caps::CompressedVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } = absolute_caps {
            self.dims[input] = (*w, *h);
        }
        self.kinds[input] = Some(kind);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
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
                    if let MemoryDomain::System(s) = &frame.domain {
                        let au = s.as_slice().to_vec();
                        self.capture_init(input, &au);
                    }
                    self.agg.push(input, frame);
                }
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; the moov is
                // fixed from the first AU's in-band init.
                PipelinePacket::CapsChanged(_) => return Ok(()),
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
            Ok(())
        })
    }
}

/// Synthesise the 2-byte AAC AudioSpecificConfig from an ADTS header.
pub(crate) fn asc_from_adts(au: &[u8]) -> Option<[u8; 2]> {
    if au.len() < 7 || au[0] != 0xFF || (au[1] & 0xF0) != 0xF0 {
        return None;
    }
    let object_type = ((au[2] >> 6) & 0x03) + 1; // profile + 1
    let sr_index = (au[2] >> 2) & 0x0F;
    let channel_config = ((au[2] & 0x01) << 2) | ((au[3] >> 6) & 0x03);
    Some([
        (object_type << 3) | (sr_index >> 1),
        ((sr_index & 1) << 7) | (channel_config << 3),
    ])
}

/// Strip the ADTS header (7 bytes, or 9 with CRC) from an AAC access unit.
pub(crate) fn strip_adts(au: &[u8]) -> &[u8] {
    if au.len() >= 7 && au[0] == 0xFF && (au[1] & 0xF0) == 0xF0 {
        let header = if au[1] & 0x01 == 0 { 9 } else { 7 }; // protection_absent==0 -> CRC
        au.get(header..).unwrap_or(&[])
    } else {
        au
    }
}

fn ns_to_ts(ns: u64, timescale: u32) -> u64 {
    (ns as u128 * timescale as u128 / 1_000_000_000) as u64
}

/// Build a multi-track `moov`: `mvhd` + one `trak` per track + `mvex` (one
/// `trex` per track). `track_ID` is the 1-based track index.
fn av_moov(tracks: &[TrackInit]) -> Vec<u8> {
    let next_track_id = (tracks.len() + 1) as u32;
    let mvhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&1000u32.to_be_bytes()); // movie timescale
        p.extend_from_slice(&0u32.to_be_bytes()); // duration (fragmented)
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
        body.extend_from_slice(&trak(i as u32 + 1, track));
    }
    let mvex = {
        let mut p = Vec::new();
        for i in 0..tracks.len() {
            let mut t = Vec::new();
            t.extend_from_slice(&(i as u32 + 1).to_be_bytes()); // track id
            t.extend_from_slice(&1u32.to_be_bytes()); // default sample description
            t.extend_from_slice(&[0u8; 12]); // default duration/size/flags
            p.extend_from_slice(&full_box(b"trex", 0, 0, &t));
        }
        mp4_box(b"mvex", &p)
    };
    body.extend_from_slice(&mvex);
    mp4_box(b"moov", &body)
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
        TrackInit::Video { codec, width, height, param_sets } => {
            let refs: Vec<&[u8]> = param_sets.iter().map(|v| v.as_slice()).collect();
            TrakMedia {
                handler: b"vide",
                media_header: full_box(b"vmhd", 0, 1, &[0u8; 8]),
                sample_entry: visual_sample_entry(*codec, *width, *height, &refs),
                timescale: VIDEO_TIMESCALE,
                dims: (*width, *height),
                is_video: true,
            }
        }
        TrackInit::Audio { format, channels, rate, asc } => {
            let sample_entry = match format {
                AudioFormat::Opus => audio_sample_entry(b"Opus", *channels, *rate, &dops(*channels, *rate)),
                _ => audio_sample_entry(b"mp4a", *channels, *rate, &esds(asc)),
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
/// `OpusHead`); channel mapping family 0 (mono/stereo), a conventional 80 ms
/// pre-skip (the exact encoder delay is not surfaced in caps).
fn dops(channels: u8, rate: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(0); // Version
    b.push(channels.max(1)); // OutputChannelCount
    b.extend_from_slice(&3840u16.to_be_bytes()); // PreSkip
    b.extend_from_slice(&rate.to_be_bytes()); // InputSampleRate
    b.extend_from_slice(&0i16.to_be_bytes()); // OutputGain
    b.push(0); // ChannelMappingFamily
    mp4_box(b"dOps", &b)
}

/// One `trak` for a track (`track_ID` 1-based).
fn trak(track_id: u32, init: &TrackInit) -> Vec<u8> {
    let TrakMedia { handler, media_header: header, sample_entry, timescale, dims, is_video } =
        trak_media(init);
    let tkhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]); // times
        p.extend_from_slice(&track_id.to_be_bytes());
        p.extend_from_slice(&[0u8; 4]); // reserved
        p.extend_from_slice(&0u32.to_be_bytes()); // duration
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
        p.extend_from_slice(&0u32.to_be_bytes()); // duration
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
        let empty4 = 0u32.to_be_bytes();
        let stts = full_box(b"stts", 0, 0, &empty4);
        let stsc = full_box(b"stsc", 0, 0, &empty4);
        let stsz = full_box(b"stsz", 0, 0, &[0u8; 8]);
        let stco = full_box(b"stco", 0, 0, &empty4);
        mp4_box(b"stbl", &[stsd, stts, stsc, stsz, stco].concat())
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
    mp4_box(b"trak", &[tkhd, mdia].concat())
}

/// One `moof`+`mdat` fragment holding a single sample for `track_id`.
fn av_fragment(track_id: u32, sequence: u64, decode_time: u64, duration: u32, sample: &[u8], is_sync: bool) -> Vec<u8> {
    let sample_flags: u32 = if is_sync { 0x0200_0000 } else { 0x0101_0000 };
    let build_moof = |data_offset: u32| -> Vec<u8> {
        let mfhd = full_box(b"mfhd", 0, 0, &(sequence as u32).to_be_bytes());
        let tfhd = full_box(b"tfhd", 0, 0x020000, &track_id.to_be_bytes()); // default-base-is-moof
        let tfdt = full_box(b"tfdt", 1, 0, &decode_time.to_be_bytes());
        let trun = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes()); // sample count
            p.extend_from_slice(&data_offset.to_be_bytes());
            p.extend_from_slice(&duration.to_be_bytes());
            p.extend_from_slice(&(sample.len() as u32).to_be_bytes());
            p.extend_from_slice(&sample_flags.to_be_bytes());
            full_box(b"trun", 0, 0x000701, &p) // data-offset | duration | size | flags
        };
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        mp4_box(b"moof", &[mfhd, traf].concat())
    };
    let moof_len = build_moof(0).len() as u32;
    let moof = build_moof(moof_len + 8);
    let mdat = mp4_box(b"mdat", sample);
    [moof, mdat].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

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
            TrackInit::Video {
                codec: VideoCodec::H264,
                width: 320,
                height: 240,
                param_sets: alloc::vec![alloc::vec![0x67, 0x42, 0x00, 0x1e], alloc::vec![0x68, 0xce]],
            },
            TrackInit::Audio {
                format: AudioFormat::Aac,
                channels: 2,
                rate: 48000,
                asc: alloc::vec![0x11, 0x90],
            },
        ];
        let moov = av_moov(&tracks);
        let count = |needle: &[u8]| moov.windows(4).filter(|w| *w == needle).count();
        assert_eq!(count(b"trak"), 2, "one trak per track");
        assert_eq!(count(b"trex"), 2, "one trex per track");
        assert_eq!(count(b"avcC"), 1, "video sample entry");
        assert_eq!(count(b"esds"), 1, "audio sample entry");
        assert_eq!(count(b"soun"), 1);
        assert_eq!(count(b"vide"), 1);
    }

    #[test]
    fn opus_track_writes_an_opus_sample_entry_with_dops() {
        let tracks = [TrackInit::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            rate: 48000,
            asc: Vec::new(),
        }];
        let moov = av_moov(&tracks);
        let count = |needle: &[u8]| moov.windows(needle.len()).filter(|w| *w == needle).count();
        assert_eq!(count(b"Opus"), 1, "Opus sample entry");
        assert_eq!(count(b"dOps"), 1, "OpusSpecificBox");
        assert_eq!(count(b"esds"), 0, "no AAC descriptor for an Opus track");
        assert_eq!(count(b"soun"), 1, "sound handler");
    }
}

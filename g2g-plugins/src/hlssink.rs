//! HLS packaging sink (`hlssink`). Cuts the muxed byte stream arriving on its
//! sink pad into media segment files and publishes a rolling `.m3u8` media
//! playlist beside them, so a standard HLS client can play the output live or,
//! after `#EXT-X-ENDLIST`, as VOD:
//!
//! ```text
//! ... ! tsmux ! hlssink location=seg%05d.ts playlist-location=out.m3u8
//! ... ! mp4mux fragment-duration=2000 ! hlssink location=seg%05d.m4s init-location=init.mp4
//! ```
//!
//! The muxer stays a separate element (GStreamer's `hlssink2` bundles one
//! internally), so the same sink packages MPEG-TS or CMAF / fMP4 depending on
//! what feeds it.
//!
//! A segment may only start at a keyframe, and closes at the first keyframe at
//! or past `target-duration` (`0` cuts at every keyframe and leaves the pacing to
//! the muxer). The two containers say "keyframe" differently:
//!
//! - MPEG-TS: one input frame is one access unit, so `FrameTiming::keyframe`
//!   marks the boundary candidates and the frame PTSs give the durations.
//! - fMP4: the byte stream is walked as boxes. A `moof` whose first sample is a
//!   sync sample opens a fragment and is a boundary candidate; the `trun` sample
//!   durations in the track timescale give the exact segment duration. `ftyp` +
//!   `moov` are the init segment: written once to `init-location` and named by
//!   the playlist's `#EXT-X-MAP`.
//!
//! Playlist URIs are the *basename* of the segment path (prefixed by
//! `playlist-root` when set), so the playlist, its segments, and the init
//! segment are expected to live in one directory.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use std::fs;
use std::io::Write;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::filesink::{io_err, path_io_err};
use crate::fmp4::{parse_trun, tfhd_defaults, trun_first_sample_is_sync};
use crate::hls::{write_media, MediaPlaylist, Segment};
use crate::mp4box::{be32, boxes, find_box, find_path};
use crate::multifilesink::expand;
use g2g_core::log::short_type_name;

/// # Example
///
/// ```no_run
/// use g2g_plugins::hlssink::HlsSink;
///
/// // ... ! tsmux ! hlssink location=seg%05d.ts playlist-location=out.m3u8
/// let sink = HlsSink::new("seg%05d.ts")
///     .with_playlist_location("out.m3u8")
///     .with_target_duration(4);
/// ```
pub struct HlsSink {
    location: String,
    playlist_location: String,
    init_location: String,
    playlist_root: String,
    target_duration_secs: u64,
    playlist_length: u64,
    max_files: u64,
    /// Which container the muxer upstream is producing, from the negotiated caps.
    encoding: Option<ByteStreamEncoding>,
    /// Index of the next segment file (the `location` pattern's printf field).
    index: u64,
    /// Bytes of the segment being cut, and its duration so far.
    open_segment: Vec<u8>,
    open_duration_ns: u64,
    /// fMP4 only: the init segment (`ftyp`+`moov`), the media timescale per
    /// track it declares, and the boxes (`styp` / `prft`) that belong ahead of
    /// the next `moof` rather than to the segment already open.
    init: Vec<u8>,
    init_written: bool,
    timescales: Vec<(u32, u32)>,
    pending_header: Vec<u8>,
    /// MPEG-TS only: the PTS the open segment starts at, the last PTS seen, and
    /// the last inter-frame delta (the final frame's duration is not carried, so
    /// the previous delta stands in for it).
    segment_start_pts_ns: u64,
    last_pts_ns: Option<u64>,
    frame_duration_ns: u64,
    playlist: MediaPlaylist,
    /// Segment files on disk in write order, for `max-files` pruning.
    on_disk: VecDeque<String>,
    segments_written: u64,
}

impl core::fmt::Debug for HlsSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HlsSink")
            .field("location", &self.location)
            .field("playlist_location", &self.playlist_location)
            .field("target_duration_secs", &self.target_duration_secs)
            .field("playlist_length", &self.playlist_length)
            .field("max_files", &self.max_files)
            .field("segments_written", &self.segments_written)
            .finish_non_exhaustive()
    }
}

impl Default for HlsSink {
    fn default() -> Self {
        Self::new("segment%05d.ts")
    }
}

impl HlsSink {
    /// `location` is a printf-style segment pattern with one integer field, e.g.
    /// `segment%05d.ts`; without a field the index is appended.
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            playlist_location: String::from("playlist.m3u8"),
            init_location: String::from("init.mp4"),
            playlist_root: String::new(),
            target_duration_secs: 15,
            playlist_length: 5,
            max_files: 10,
            encoding: None,
            index: 0,
            open_segment: Vec::new(),
            open_duration_ns: 0,
            init: Vec::new(),
            init_written: false,
            timescales: Vec::new(),
            pending_header: Vec::new(),
            segment_start_pts_ns: 0,
            last_pts_ns: None,
            frame_duration_ns: 0,
            playlist: MediaPlaylist {
                target_duration_secs: 0,
                media_sequence: 0,
                segments: Vec::new(),
                map_uri: None,
                map_byte_range: None,
                end_list: false,
                part_target_ms: None,
                server_control: None,
            },
            on_disk: VecDeque::new(),
            segments_written: 0,
        }
    }

    /// Where the `.m3u8` media playlist is written.
    pub fn with_playlist_location(mut self, path: impl Into<String>) -> Self {
        self.playlist_location = path.into();
        self
    }

    /// Where the fMP4 init segment (`ftyp`+`moov`) is written. Unused for
    /// MPEG-TS, which carries its PAT/PMT in every segment.
    pub fn with_init_location(mut self, path: impl Into<String>) -> Self {
        self.init_location = path.into();
        self
    }

    /// Prefix prepended to each segment URI in the playlist, for serving the
    /// segments from a different path than the playlist.
    pub fn with_playlist_root(mut self, root: impl Into<String>) -> Self {
        self.playlist_root = root.into();
        self
    }

    /// Target segment duration in seconds; a segment closes at the first keyframe
    /// at or past it. `0` cuts at every keyframe.
    pub fn with_target_duration(mut self, secs: u64) -> Self {
        self.target_duration_secs = secs;
        self
    }

    /// How many segments the playlist lists (`0` = every segment ever written,
    /// the VOD case). Dropping the oldest advances `#EXT-X-MEDIA-SEQUENCE`.
    pub fn with_playlist_length(mut self, count: u64) -> Self {
        self.playlist_length = count;
        self
    }

    /// How many segment files to keep on disk (`0` = keep all).
    pub fn with_max_files(mut self, count: u64) -> Self {
        self.max_files = count;
        self
    }

    /// Segments closed and written so far.
    pub fn segments_written(&self) -> u64 {
        self.segments_written
    }

    /// The playlist as published, for tests and for a caller serving it from
    /// memory.
    pub fn playlist(&self) -> &MediaPlaylist {
        &self.playlist
    }

    /// The URI a segment path gets in the playlist: its basename, under
    /// `playlist-root` when one is set.
    fn playlist_uri(&self, path: &str) -> String {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        if self.playlist_root.is_empty() {
            return String::from(name);
        }
        let sep = if self.playlist_root.ends_with('/') {
            ""
        } else {
            "/"
        };
        alloc::format!("{}{sep}{name}", self.playlist_root)
    }

    /// Whether the open segment has reached the target, i.e. the next keyframe
    /// should start a new one. An empty segment never cuts (that would write a
    /// zero-length file at the very first keyframe).
    fn segment_due(&self) -> bool {
        !self.open_segment.is_empty()
            && self.open_duration_ns >= self.target_duration_secs.saturating_mul(1_000_000_000)
    }

    /// Write the open segment out, list it, and republish the playlist.
    fn close_segment(&mut self) -> Result<(), G2gError> {
        if self.open_segment.is_empty() {
            return Ok(());
        }
        if !self.init.is_empty() && !self.init_written {
            fs::write(&self.init_location, &self.init).map_err(|e| {
                path_io_err(short_type_name::<Self>(), "write", &self.init_location, e)
            })?;
            self.playlist.map_uri = Some(self.playlist_uri(&self.init_location));
            self.init_written = true;
        }
        let path = expand(&self.location, self.index);
        let mut file = fs::File::create(&path)
            .map_err(|e| path_io_err(short_type_name::<Self>(), "create", &path, e))?;
        file.write_all(&self.open_segment).map_err(io_err)?;
        file.flush().map_err(io_err)?;
        self.index += 1;
        self.segments_written += 1;
        self.on_disk.push_back(path.clone());

        self.playlist.segments.push(Segment {
            uri: self.playlist_uri(&path),
            duration_ms: (self.open_duration_ns / 1_000_000) as u32,
            key: None,
            byte_range: None,
            gap: false,
            parts: Vec::new(),
        });
        self.open_segment.clear();
        self.open_duration_ns = 0;

        if self.playlist_length > 0 {
            while self.playlist.segments.len() as u64 > self.playlist_length {
                self.playlist.segments.remove(0);
                self.playlist.media_sequence += 1;
            }
        }
        // RFC 8216 6.3.3: EXT-X-TARGETDURATION must be at least the longest
        // listed segment, rounded to seconds, and may never be 0.
        let longest = self
            .playlist
            .segments
            .iter()
            .map(|s| s.duration_ms.div_ceil(1000) as u64)
            .max()
            .unwrap_or(0);
        self.playlist.target_duration_secs = self
            .target_duration_secs
            .max(longest)
            .max(1)
            .min(u32::MAX as u64) as u32;

        while self.max_files > 0 && self.on_disk.len() as u64 > self.max_files {
            if let Some(old) = self.on_disk.pop_front() {
                let _ = fs::remove_file(old);
            }
        }
        self.write_playlist()
    }

    fn write_playlist(&self) -> Result<(), G2gError> {
        fs::write(&self.playlist_location, write_media(&self.playlist)).map_err(|e| {
            path_io_err(
                short_type_name::<Self>(),
                "write",
                &self.playlist_location,
                e,
            )
        })
    }

    /// One MPEG-TS input frame: a whole access unit's packets. The keyframe flag
    /// says whether it may open a segment.
    fn push_ts(&mut self, bytes: &[u8], pts_ns: u64, keyframe: bool) -> Result<(), G2gError> {
        if keyframe && self.segment_due() {
            self.close_segment()?;
        }
        if self.open_segment.is_empty() {
            self.segment_start_pts_ns = pts_ns;
        }
        if let Some(prev) = self.last_pts_ns {
            if pts_ns > prev {
                self.frame_duration_ns = pts_ns - prev;
            }
        }
        self.last_pts_ns = Some(pts_ns);
        self.open_segment.extend_from_slice(bytes);
        // Through the end of this frame: its own duration is only known once the
        // next one arrives, so the previous delta stands in for it.
        self.open_duration_ns = pts_ns
            .saturating_sub(self.segment_start_pts_ns)
            .saturating_add(self.frame_duration_ns);
        Ok(())
    }

    /// One fMP4 input frame: whole top-level boxes (an init segment, a fragment,
    /// or a CMAF chunk). A frame that is not a whole number of boxes is a
    /// malformed stream and fails the push.
    fn push_bmff(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let mut consumed = 0usize;
        for (kind, payload) in boxes(bytes) {
            let whole = &bytes[consumed..consumed + payload.len() + 8];
            consumed += whole.len();
            match kind {
                b"ftyp" | b"moov" => {
                    if kind == b"moov" {
                        self.timescales = track_timescales(payload)?;
                    }
                    self.init.extend_from_slice(whole);
                }
                // The segment header sits ahead of the `moof` it opens, so it is
                // held until that `moof` has decided where the cut goes.
                b"styp" | b"prft" => self.pending_header.extend_from_slice(whole),
                b"moof" => {
                    let (track_id, sync, ticks) = fragment_info(payload, bytes.len())?;
                    if sync && self.segment_due() {
                        self.close_segment()?;
                    }
                    self.open_segment.append(&mut self.pending_header);
                    self.open_segment.extend_from_slice(whole);
                    let timescale = self
                        .timescales
                        .iter()
                        .find(|(id, _)| *id == track_id)
                        .map(|(_, ts)| *ts)
                        .ok_or(G2gError::CapsMismatch)?;
                    self.open_duration_ns = self
                        .open_duration_ns
                        .saturating_add((ticks as u128 * 1_000_000_000 / timescale as u128) as u64);
                }
                _ => self.open_segment.extend_from_slice(whole),
            }
        }
        if consumed != bytes.len() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(())
    }
}

/// A fragment's (track id, whether its first sample is a sync sample, total
/// sample duration in the track timescale), read from the first `traf`.
fn fragment_info(moof: &[u8], frame_len: usize) -> Result<(u32, bool, u64), G2gError> {
    let traf = find_box(moof, b"traf").ok_or(G2gError::CapsMismatch)?;
    let tfhd = find_box(traf, b"tfhd").ok_or(G2gError::CapsMismatch)?;
    let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
    let track_id = be32(tfhd, 4)?;
    let (default_duration, default_size) = tfhd_defaults(tfhd)?;
    let (_, durations) = parse_trun(trun, default_duration, default_size, frame_len)?;
    let ticks = durations.iter().map(|d| *d as u64).sum();
    Ok((track_id, trun_first_sample_is_sync(trun)?, ticks))
}

/// Each track's (track id, media timescale) from a `moov`. Only the two header
/// boxes are read: unlike the demuxer's header parse this must not care whether
/// the sample entry names a codec we can decode.
fn track_timescales(moov: &[u8]) -> Result<Vec<(u32, u32)>, G2gError> {
    let mut out = Vec::new();
    for (kind, trak) in boxes(moov) {
        if kind != b"trak" {
            continue;
        }
        let tkhd = find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?;
        // tkhd v0: track_ID at payload offset 12 (4 version/flags + 8 times).
        let track_id = be32(tkhd, 12)?;
        let mdhd = find_path(trak, &[b"mdia", b"mdhd"]).ok_or(G2gError::CapsMismatch)?;
        // mdhd v0: timescale at payload offset 12.
        let timescale = be32(mdhd, 12)?;
        if tkhd.first() != Some(&0) || mdhd.first() != Some(&0) || timescale == 0 {
            return Err(G2gError::CapsMismatch);
        }
        out.push((track_id, timescale));
    }
    if out.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(out)
}

fn accepted_caps() -> Vec<Caps> {
    Vec::from([
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
    ])
}

impl AsyncElement for HlsSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        encoding_of(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(accepted_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.encoding = Some(encoding_of(absolute_caps)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    match self.encoding.ok_or(G2gError::NotConfigured)? {
                        ByteStreamEncoding::MpegTs => {
                            self.push_ts(slice, frame.timing.pts_ns, frame.timing.keyframe)?
                        }
                        ByteStreamEncoding::IsoBmff => self.push_bmff(slice)?,
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Eos => {
                    self.close_segment()?;
                    self.playlist.end_list = true;
                    self.write_playlist()?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.encoding = Some(encoding_of(&c)?);
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HLSSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HLS sink",
            "Sink/File",
            "Segments a muxed byte stream and writes an HLS media playlist",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => self.location = value.as_str().ok_or(PropError::Type)?.into(),
            "playlist-location" => {
                self.playlist_location = value.as_str().ok_or(PropError::Type)?.into()
            }
            "init-location" => self.init_location = value.as_str().ok_or(PropError::Type)?.into(),
            "playlist-root" => self.playlist_root = value.as_str().ok_or(PropError::Type)?.into(),
            "target-duration" => {
                self.target_duration_secs = value.as_uint().ok_or(PropError::Type)?
            }
            "playlist-length" => self.playlist_length = value.as_uint().ok_or(PropError::Type)?,
            "max-files" => self.max_files = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "playlist-location" => Some(PropValue::Str(self.playlist_location.clone())),
            "init-location" => Some(PropValue::Str(self.init_location.clone())),
            "playlist-root" => Some(PropValue::Str(self.playlist_root.clone())),
            "target-duration" => Some(PropValue::Uint(self.target_duration_secs)),
            "playlist-length" => Some(PropValue::Uint(self.playlist_length)),
            "max-files" => Some(PropValue::Uint(self.max_files)),
            _ => None,
        }
    }
}

/// The container a byte-stream caps names, rejecting anything this sink cannot
/// segment.
fn encoding_of(caps: &Caps) -> Result<ByteStreamEncoding, G2gError> {
    match caps {
        Caps::ByteStream {
            encoding: encoding @ (ByteStreamEncoding::MpegTs | ByteStreamEncoding::IsoBmff),
        } => Ok(*encoding),
        _ => Err(G2gError::CapsMismatch),
    }
}

static HLSSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "printf-style segment file pattern, e.g. segment%05d.ts",
    ),
    PropertySpec::new(
        "playlist-location",
        PropKind::Str,
        "path of the .m3u8 media playlist",
    ),
    PropertySpec::new(
        "init-location",
        PropKind::Str,
        "path of the fMP4 init segment named by #EXT-X-MAP",
    ),
    PropertySpec::new(
        "playlist-root",
        PropKind::Str,
        "prefix for the segment URIs written into the playlist",
    ),
    PropertySpec::new(
        "target-duration",
        PropKind::Uint,
        "target segment duration in seconds (0 = cut at every keyframe)",
    ),
    PropertySpec::new(
        "playlist-length",
        PropKind::Uint,
        "segments listed in the playlist (0 = all)",
    ),
    PropertySpec::new(
        "max-files",
        PropKind::Uint,
        "segment files kept on disk (0 = all)",
    ),
];

impl PadTemplates for HlsSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(
            accepted_caps(),
        ))])
    }
}

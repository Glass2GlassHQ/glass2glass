//! Fragmented-MP4 demuxer source (M28, HEVC in M31), the read-side counterpart
//! of `Mp4Mux`: parses a single-video-track fMP4 and emits Annex-B H.264 or
//! H.265 access units with their recovered timing, so a recording plays back
//! through `MfDecode` / `FfmpegH264Dec` exactly like a live stream.
//!
//! Caps discovery is the M18 async-source path: `intercept_caps` reads the
//! file's `ftyp`/`moov` (dims from `tkhd`, codec + parameter sets from the
//! `avc1`/`avcC` or `hvc1`/`hvcC` sample entry, timescale from `mdhd`) before
//! negotiation, so downstream solves against the real geometry. The fragment
//! scan happens in `run`.
//!
//! Supported profile: what `Mp4Mux` writes and CMAF-style single-track
//! files generally share: one video track, `trun` v0 with explicit sample
//! sizes, `default-base-is-moof` data offsets landing on the following
//! `mdat`'s payload. Anything else fails loud rather than emitting a
//! corrupt bitstream. If the first sample carries no in-band parameter sets,
//! the ones from the config record are prepended so a decoder can start.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    BusHandle, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, Rate, Segment,
};

use crate::filesink::io_err;
use crate::fmp4::{parse_fragments, parse_header, starts_with_param_set, Header, Sample};
use crate::mp4box::{find_box, parse_ilst_tags};

#[derive(Debug)]
pub struct Mp4Src {
    path: PathBuf,
    header: Option<Header>,
    configured: bool,
    bus: Option<BusHandle>,
    seek: Option<SeekController>,
}

impl Mp4Src {
    /// The file is read during caps probing and `run`; construction has no
    /// filesystem side effects.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: None,
            configured: false,
            bus: None,
            seek: None,
        }
    }

    /// Attach the pipeline bus so the file's `moov/udta/meta/ilst` metadata posts
    /// as a [`BusMessage::Tag`] once read.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the source seekable: `run` polls `controller` between frames and, on a
    /// flushing seek, emits `Flush`, repositions to the keyframe at or before the
    /// target, emits the post-flush `Segment`, and resumes. The application keeps a
    /// clone of the controller to drive scrubbing / editing.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.header.is_none() {
            let data = std::fs::read(&self.path).map_err(io_err)?;
            self.header = Some(parse_header(&data)?);
        }
        let h = self.header.as_ref().expect("just parsed");
        Ok(Caps::CompressedVideo {
            codec: h.codec,
            width: Dim::Fixed(h.width),
            height: Dim::Fixed(h.height),
            framerate: Rate::Any,
        })
    }
}

impl SourceLoop for Mp4Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    /// Header probe during negotiation (file I/O is synchronous, so a
    /// ready future carries the result).
    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.probe())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.probe()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// The movie duration parsed from `mdhd` (M203), known after the header
    /// probe at negotiation. `None` for a file whose box reports `0`. The runner
    /// publishes it on the progress handle and posts `DurationChanged`.
    fn query_duration(&self) -> Option<u64> {
        self.header.as_ref().and_then(|h| h.duration_ns)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let data = std::fs::read(&self.path).map_err(io_err)?;
            if self.header.is_none() {
                self.header = Some(parse_header(&data)?);
            }
            // Surface the file's metadata once, before the samples flow.
            if let Some(bus) = &self.bus {
                if let Some(moov) = find_box(&data, b"moov") {
                    let tags = parse_ilst_tags(moov);
                    if !tags.is_empty() {
                        bus.try_post(BusMessage::Tag(tags));
                    }
                }
            }
            let header = self.header.as_ref().expect("parsed above");
            // No decryptor here: an encrypted (cbcs) file fails loud rather than
            // emitting garbage. Decryption lives in `fmp4demux` (the HLS path).
            let samples =
                parse_fragments(&data, header.timescale, header.codec, header.cenc.as_ref(), None)?;

            let mut sequence = 0u64;
            // The next emitted frame is a (re)start: prepend the out-of-band
            // parameter sets if it lacks them, so a decoder can resume. Set again
            // after every seek, since the landed keyframe also needs them.
            let mut need_param_sets = true;
            let mut i = 0usize;
            while i < samples.len() {
                // A flushing seek repositions to the keyframe at or before the
                // target before the next frame is produced (GStreamer-style:
                // upstream to the source, latest-wins).
                if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                    if seek.is_flush() {
                        out.push(PipelinePacket::Flush).await?;
                        i = keyframe_index_for(&samples, seek.start);
                        need_param_sets = true;
                        out.push(PipelinePacket::Segment(Segment::for_flush_seek(&seek, None)))
                            .await?;
                    }
                    continue; // re-evaluate from the repositioned index
                }

                let s = &samples[i];
                let mut annexb = s.annexb.clone();
                if need_param_sets && !starts_with_param_set(&annexb, header.codec) {
                    // out-of-band parameter sets: prepend so a decoder can
                    // start (our own writer keeps them in-band).
                    let mut with_sets = Vec::new();
                    for set in &header.param_sets {
                        with_sets.extend_from_slice(&[0, 0, 0, 1]);
                        with_sets.extend_from_slice(set);
                    }
                    with_sets.extend_from_slice(&annexb);
                    annexb = with_sets;
                }
                need_param_sets = false;
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
                        keyframe: s.keyframe, // MP4 sync-sample (stss) table
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                i += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

/// The index of the keyframe at or before `target_ns` (GStreamer `SNAP_BEFORE`,
/// so a decoder can resume from a clean reference); 0 when none precedes it.
fn keyframe_index_for(samples: &[Sample], target_ns: u64) -> usize {
    samples
        .iter()
        .enumerate()
        .rfind(|(_, s)| s.keyframe && s.pts_ns <= target_ns)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

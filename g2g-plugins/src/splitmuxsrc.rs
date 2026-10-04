//! Split-muxer source (`splitmuxsrc`): plays the parts a `splitmuxsink` wrote,
//! in name order, as one continuous elementary stream with a single `Eos`.
//!
//! Each part is its own container, so each is demuxed on its own: an MP4 part
//! through `Mp4Src`, a Matroska or MPEG-TS part through that container's demuxer
//! selecting the part's primary stream. The first part plays at its own
//! timestamps. Every later part is moved so its first frame lands where the
//! previous part ended, whether its muxer restarted the clock at zero or kept
//! counting, and its stream-start `Segment` is dropped so the outer stream keeps
//! the first part's.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::fs::File;
use std::path::{Path, PathBuf};

use g2g_core::element::DynAsyncElement;
use g2g_core::log::{short_type_name, Target};
use g2g_core::runtime::{PrimaryStreamHook, SourceLoop};
use g2g_core::{
    g2g_error, ByteStreamEncoding, Caps, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, PushOutcome,
};

use crate::filesink::path_io_err;
use crate::filesrc::sniff_file_caps;
use crate::gaplesssrc::{shift_packet, Shifted, INNER_TIMELINE_START};
use crate::mkvdemux::MkvDemux;
use crate::mp4src::Mp4Src;
use crate::splitfilesrc::{matching_parts, read_block, DEFAULT_BLOCKSIZE};
use crate::tsdemux::TsDemux;
use crate::uridecodebin::{mkv_primary_stream, ts_primary_stream};

static SPLITMUXSRC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "wildcard pattern matching the parts, e.g. clip*.mp4, matched on the file name and played in name order",
)];

type DemuxConstructor = fn() -> Box<dyn DynAsyncElement>;

// the containers a part is demuxed from bytes, with the hook that picks its stream
static DEMUXED_CONTAINERS: &[(ByteStreamEncoding, PrimaryStreamHook, DemuxConstructor)] = &[
    (ByteStreamEncoding::Matroska, mkv_primary_stream, || {
        Box::new(MkvDemux::new())
    }),
    (ByteStreamEncoding::MpegTs, ts_primary_stream, || {
        Box::new(TsDemux::new())
    }),
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::splitmuxsrc::SplitMuxSrc;
///
/// // gst-launch equivalent: splitmuxsrc location="clip*.mp4" ! h264parse ! ...
/// let src = SplitMuxSrc::new("clip*.mp4");
/// ```
#[derive(Debug)]
pub struct SplitMuxSrc {
    location: String,
    parts: Vec<PathBuf>,
    // the first part's stream, which every later part has to match
    stream_caps: Option<Caps>,
    configured: bool,
}

impl SplitMuxSrc {
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            parts: Vec::new(),
            stream_caps: None,
            configured: false,
        }
    }

    fn resolve(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.stream_caps {
            return Ok(caps.clone());
        }
        let parts = matching_parts(&self.location, short_type_name::<Self>())?;
        let Some(first) = parts.first() else {
            g2g_error!(
                Target::category(short_type_name::<Self>()),
                "no file matches {}",
                self.location
            );
            return Err(G2gError::CapsMismatch);
        };
        let caps = Part::open(first)?.caps;
        self.parts = parts;
        self.stream_caps = Some(caps.clone());
        Ok(caps)
    }
}

enum PartReader {
    // an ISO-BMFF part demuxes itself
    SelfDemuxing(Mp4Src),
    Demuxed {
        path: PathBuf,
        container: Caps,
        demux: Box<dyn DynAsyncElement>,
    },
}

struct Part {
    reader: PartReader,
    caps: Caps,
}

impl Part {
    fn open(path: &Path) -> Result<Self, G2gError> {
        let container = sniff_file_caps(path, short_type_name::<SplitMuxSrc>())?;
        let Caps::ByteStream { encoding } = container else {
            return Err(unsupported_part(path, &container));
        };
        if matches!(
            encoding,
            ByteStreamEncoding::Mp4 | ByteStreamEncoding::IsoBmff
        ) {
            let mut source = Mp4Src::new(path);
            let caps = SourceLoop::intercept_caps(&mut source).into_inner()?;
            return Ok(Self {
                reader: PartReader::SelfDemuxing(source),
                caps,
            });
        }
        let Some((_, primary_stream, construct)) = DEMUXED_CONTAINERS
            .iter()
            .find(|(demuxed, _, _)| *demuxed == encoding)
        else {
            return Err(unsupported_part(path, &container));
        };
        let location = path.to_string_lossy();
        let Some(primary) = primary_stream(&location, &container) else {
            return Err(unsupported_part(path, &container));
        };
        let mut demux = construct();
        for (name, value) in primary.props {
            demux
                .set_property(&name, PropValue::Str(value))
                .map_err(|_| unsupported_part(path, &container))?;
        }
        Ok(Self {
            reader: PartReader::Demuxed {
                path: path.to_path_buf(),
                container,
                demux,
            },
            caps: primary.caps,
        })
    }

    async fn play(self, out: &mut PartSink<'_>) -> Result<(), G2gError> {
        match self.reader {
            PartReader::SelfDemuxing(mut source) => {
                SourceLoop::configure_pipeline(&mut source, &self.caps)?;
                SourceLoop::run(&mut source, out).await?;
            }
            PartReader::Demuxed {
                path,
                container,
                mut demux,
            } => {
                demux.configure_pipeline(&container)?;
                let mut file = File::open(&path)
                    .map_err(|e| path_io_err(short_type_name::<SplitMuxSrc>(), "open", &path, e))?;
                let mut sequence = 0u64;
                while let Some(frame) = read_block(&mut file, DEFAULT_BLOCKSIZE, sequence)? {
                    sequence += 1;
                    demux.process(PipelinePacket::DataFrame(frame), out).await?;
                }
                demux.process(PipelinePacket::Eos, out).await?;
            }
        }
        Ok(())
    }
}

fn unsupported_part(path: &Path, container: &Caps) -> G2gError {
    g2g_error!(
        Target::category(short_type_name::<SplitMuxSrc>()),
        "{} is {container:?}, not an MP4, Matroska or MPEG-TS part with a stream to play",
        path.display()
    );
    G2gError::CapsMismatch
}

// a decode chain built for `first` can take `part` without being rebuilt
fn same_stream(first: &Caps, part: &Caps) -> bool {
    match (first, part) {
        (Caps::CompressedVideo { codec: a, .. }, Caps::CompressedVideo { codec: b, .. }) => a == b,
        (Caps::Audio { format: a, .. }, Caps::Audio { format: b, .. }) => a == b,
        _ => false,
    }
}

// what one part hands the next
#[derive(Debug, Default)]
struct Timeline {
    // highest pts + duration sent so far, where the next part starts
    end: u64,
    // smallest gap between neighbouring frames, the duration of a frame that declares none
    frame_spacing: Option<u64>,
    caps: Option<Caps>,
    // frames sent so far, the next frame's sequence number
    frames: u64,
}

struct PartSink<'o> {
    out: &'o mut dyn OutputSink,
    timeline: &'o mut Timeline,
    continues_timeline: bool,
    // `None` until the first frame of a continuing part names it
    origin: Option<u64>,
    offset: u64,
    previous_pts: Option<u64>,
    // the packet in the caller's slot is already rebased, so a re-poll leaves it alone
    decided: bool,
}

impl<'o> PartSink<'o> {
    fn new(out: &'o mut dyn OutputSink, timeline: &'o mut Timeline, first_part: bool) -> Self {
        let (origin, offset) = match first_part {
            true => (Some(INNER_TIMELINE_START), INNER_TIMELINE_START),
            false => (None, timeline.end),
        };
        Self {
            out,
            timeline,
            continues_timeline: !first_part,
            origin,
            offset,
            previous_pts: None,
            decided: false,
        }
    }

    fn drops(&self, packet: &Option<PipelinePacket>) -> bool {
        match packet {
            Some(PipelinePacket::Segment(_)) => self.continues_timeline,
            Some(PipelinePacket::CapsChanged(caps)) => self.timeline.caps.as_ref() == Some(caps),
            _ => false,
        }
    }

    fn record_frame(&mut self, packet: &mut Option<PipelinePacket>) {
        let Some(PipelinePacket::DataFrame(frame)) = packet else {
            return;
        };
        // every part numbers its frames from zero
        frame.sequence = self.timeline.frames;
        let pts = frame.timing.pts_ns;
        if let Some(previous) = self.previous_pts.replace(pts) {
            let gap = pts.abs_diff(previous);
            if gap > 0 {
                let spacing = self.timeline.frame_spacing.map_or(gap, |s| s.min(gap));
                self.timeline.frame_spacing = Some(spacing);
            }
        }
        let duration = match frame.timing.duration_ns {
            0 => self.timeline.frame_spacing.unwrap_or_default(),
            declared => declared,
        };
        self.timeline.end = self.timeline.end.max(pts.saturating_add(duration));
        self.timeline.frames = self.timeline.frames.saturating_add(1);
    }
}

impl OutputSink for PartSink<'_> {
    fn begin_push(&mut self) {
        self.decided = false;
        self.out.begin_push();
    }

    fn poll_push(
        &mut self,
        cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if !self.decided {
            if self.drops(packet) {
                packet.take();
                return Poll::Ready(Ok(PushOutcome::Accepted));
            }
            match packet {
                Some(PipelinePacket::CapsChanged(caps)) => self.timeline.caps = Some(caps.clone()),
                Some(PipelinePacket::DataFrame(frame)) if self.origin.is_none() => {
                    self.origin = Some(frame.timing.pts_ns);
                }
                _ => {}
            }
            let origin = self.origin.unwrap_or(INNER_TIMELINE_START);
            match shift_packet(packet, origin, self.offset) {
                Shifted::Frame(_) => self.record_frame(packet),
                Shifted::Swallowed => return Poll::Ready(Ok(PushOutcome::Accepted)),
                Shifted::Passed => {}
            }
            self.decided = true;
        }
        self.out.poll_push(cx, packet)
    }
}

impl SourceLoop for SplitMuxSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.resolve())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.resolve()?)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn probe_output_caps(&mut self) -> Option<Caps> {
        self.resolve().ok()
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let Some(stream_caps) = self.stream_caps.clone().filter(|_| self.configured) else {
                return Err(G2gError::NotConfigured);
            };
            let mut timeline = Timeline::default();
            for (index, path) in self.parts.iter().enumerate() {
                let part = Part::open(path)?;
                if !same_stream(&stream_caps, &part.caps) {
                    return Err(unsupported_part(path, &part.caps));
                }
                let mut sink = PartSink::new(&mut *out, &mut timeline, index == 0);
                part.play(&mut sink).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(timeline.frames)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SPLITMUXSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Split-muxer source",
            "Source/File",
            "Plays the files a splitmuxsink wrote as one continuous stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.into();
                self.parts.clear();
                self.stream_caps = None;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            _ => None,
        }
    }
}

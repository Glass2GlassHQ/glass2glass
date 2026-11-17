// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, ClockTime, Target};
use g2g_core::runtime::{
    attach_seek, bounded, eos_then_wait_for_seek, flushing_only, run_graph, Receiver, Registry,
    SeekController, SeekFormat, Sender, SourceLoop,
};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    Frame, G2gError, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat, Seek, Segment,
};

use crate::clock::WallClock;
use crate::discover::discover;
use crate::edl::{parse_edl_uri, EdlSegment};
use crate::uridecodebin::PLAYBIN_MAX_DEPTH;
use crate::videofx::any_geometry;

static EDLSRC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "uri",
    PropKind::Str,
    "edl:// timeline in mpv's EDL syntax, with one stream",
)];

// matroska and millisecond-timescale mp4 store frame times rounded to whole milliseconds
const CUT_TOLERANCE_NS: u64 = 1_000_000;

#[derive(Debug)]
struct TimelineSegment {
    file: String,
    start_ns: u64,
    length_ns: u64,
    timeline_start_ns: u64,
}

impl TimelineSegment {
    fn end_ns(&self) -> u64 {
        self.start_ns.saturating_add(self.length_ns)
    }
}

fn single_stream(uri: &str) -> Result<Vec<EdlSegment>, String> {
    let mut edl = parse_edl_uri(uri).map_err(|error| error.to_string())?;
    if edl.streams.len() != 1 {
        return Err(format!(
            "the timeline has {} streams, edlsrc plays one",
            edl.streams.len()
        ));
    }
    Ok(edl.streams.remove(0).segments)
}

// mpv plays a segment with no length to the end of its file
async fn remaining_ns(registry: &Registry, segment: &EdlSegment) -> Result<u64, String> {
    let file = &segment.file;
    let discovery = discover(registry, &format!("file://{file}"))
        .await
        .map_err(|error| format!("cannot read the duration of {file}: {error}"))?;
    let duration_ns = discovery
        .duration_ns
        .ok_or_else(|| format!("{file} reports no duration"))?;
    duration_ns
        .checked_sub(segment.start_ns)
        .filter(|remaining_ns| *remaining_ns > 0)
        .ok_or_else(|| {
            format!(
                "the segment of {file} starts at {}, at or past its duration {}",
                ClockTime(segment.start_ns),
                ClockTime(duration_ns)
            )
        })
}

async fn timeline(segments: &[EdlSegment]) -> Result<Vec<TimelineSegment>, String> {
    let registry = crate::registry::default_registry();
    let mut timeline = Vec::with_capacity(segments.len());
    let mut timeline_start_ns = 0u64;
    for segment in segments {
        let length_ns = match segment.length_ns {
            Some(length_ns) => length_ns,
            None => remaining_ns(&registry, segment).await?,
        };
        timeline.push(TimelineSegment {
            file: segment.file.clone(),
            start_ns: segment.start_ns,
            length_ns,
            timeline_start_ns,
        });
        timeline_start_ns = timeline_start_ns.saturating_add(length_ns);
    }
    Ok(timeline)
}

#[derive(Debug)]
enum SegmentMessage {
    Caps(Caps),
    Frame { frame: Frame, stream_time_ns: u64 },
    End,
    Failed(G2gError),
}

fn decoded_caps() -> Caps {
    any_geometry(RawVideoFormat::Nv12)
}

#[derive(Debug)]
struct SegmentSink {
    messages: Sender<SegmentMessage>,
    segment: Segment,
    caps: Option<Caps>,
    sent_caps: Option<Caps>,
}

impl SegmentSink {
    async fn send(&self, message: SegmentMessage) -> Result<(), G2gError> {
        self.messages
            .send(message)
            .await
            .map_err(|_| G2gError::Shutdown)
    }
}

impl AsyncElement for SegmentSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(decoded_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(absolute_caps.clone());
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
                    let pts_ns = frame.timing.pts_ns;
                    let Some(stream_time_ns) = self
                        .segment
                        .presents(pts_ns)
                        .then(|| self.segment.to_stream_time(pts_ns))
                        .flatten()
                    else {
                        return Ok(());
                    };
                    let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
                    if self.sent_caps.as_ref() != Some(&caps) {
                        self.send(SegmentMessage::Caps(caps.clone())).await?;
                        self.sent_caps = Some(caps);
                    }
                    self.send(SegmentMessage::Frame {
                        frame,
                        stream_time_ns,
                    })
                    .await
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.caps = Some(caps);
                    Ok(())
                }
                PipelinePacket::Segment(segment) => {
                    self.segment = segment;
                    Ok(())
                }
                PipelinePacket::Eos => self.send(SegmentMessage::End).await,
                _ => Ok(()),
            }
        })
    }
}

async fn decode_segment(
    file: &str,
    from_ns: u64,
    messages: Sender<SegmentMessage>,
) -> Result<(), G2gError> {
    const SEGMENT_LINK_CAPACITY: usize = 2;
    let target = Target::category(short_type_name::<EdlSrc>());
    let sink = SegmentSink {
        messages,
        segment: Segment::new(),
        caps: None,
        sent_caps: None,
    };
    let is_decoded = |caps: &Caps| {
        matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            }
        )
    };
    let mut graph = crate::registry::default_registry()
        .build_uridecodebin(
            &format!("file://{file}"),
            sink,
            &is_decoded,
            PLAYBIN_MAX_DEPTH,
        )
        .map_err(|error| {
            g2g_error!(target, "cannot decode {file}: {error:?}");
            G2gError::NotConfigured
        })?;
    let (_controller, _ticket) =
        attach_seek(&mut graph, Seek::flush_to(from_ns)).map_err(|error| {
            g2g_error!(target, "cannot seek in {file}: {error:?}");
            G2gError::SeekRefused
        })?;
    run_graph(graph, &WallClock::new(), SEGMENT_LINK_CAPACITY).await?;
    Ok(())
}

#[derive(Debug)]
struct SegmentPlayback {
    timeline_index: usize,
    messages: Receiver<SegmentMessage>,
}

impl SegmentPlayback {
    fn start(
        timeline: &[TimelineSegment],
        timeline_index: usize,
        offset_ns: u64,
    ) -> Result<SegmentPlayback, G2gError> {
        const DECODED_AHEAD_FRAMES: usize = 4;
        let segment = &timeline[timeline_index];
        let from_ns = match offset_ns {
            0 => segment.start_ns.saturating_sub(CUT_TOLERANCE_NS),
            offset_ns => segment.start_ns.saturating_add(offset_ns),
        };
        let file = segment.file.clone();
        let (sender, messages) = bounded(DECODED_AHEAD_FRAMES);
        // the decode graph's elements are not `Send` in a single-threaded build, so it is built on the thread that runs it
        std::thread::Builder::new()
            .name(String::from("g2g-edl-segment"))
            .spawn(move || {
                let report = sender.clone();
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ =
                            report.try_send(SegmentMessage::Failed(g2g_core::log::io_err(error)));
                        return;
                    }
                };
                runtime.block_on(async {
                    if let Err(error) = decode_segment(&file, from_ns, sender).await {
                        let _ = report.send(SegmentMessage::Failed(error)).await;
                    }
                });
            })
            .map_err(g2g_core::log::io_err)?;
        Ok(SegmentPlayback {
            timeline_index,
            messages,
        })
    }
}

#[derive(Debug, Default)]
pub struct EdlSrc {
    uri: String,
    segments: Vec<EdlSegment>,
    timeline: Vec<TimelineSegment>,
    current: Option<SegmentPlayback>,
    next: Option<SegmentPlayback>,
    caps: Option<Caps>,
    configured: bool,
    seek: Option<SeekController>,
}

impl EdlSrc {
    pub fn new() -> Self {
        Self::default()
    }

    fn duration_ns(&self) -> u64 {
        self.timeline.last().map_or(0, |last| {
            last.timeline_start_ns.saturating_add(last.length_ns)
        })
    }

    fn play_from(&mut self, timeline_ns: u64) -> Result<(), G2gError> {
        self.current = None;
        self.next = None;
        let Some(index) = self
            .timeline
            .iter()
            .rposition(|segment| segment.timeline_start_ns <= timeline_ns)
            .filter(|_| timeline_ns < self.duration_ns())
        else {
            return Ok(());
        };
        let offset_ns = timeline_ns - self.timeline[index].timeline_start_ns;
        self.current = Some(SegmentPlayback::start(&self.timeline, index, offset_ns)?);
        self.start_next(index)
    }

    fn start_next(&mut self, current_index: usize) -> Result<(), G2gError> {
        let next_index = current_index + 1;
        if next_index < self.timeline.len() {
            self.next = Some(SegmentPlayback::start(&self.timeline, next_index, 0)?);
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<(), G2gError> {
        self.current = self.next.take();
        match &self.current {
            Some(current) => self.start_next(current.timeline_index),
            None => Ok(()),
        }
    }

    async fn first_caps(&mut self) -> Result<Caps, G2gError> {
        let target = Target::category(short_type_name::<Self>());
        if self.segments.is_empty() {
            g2g_error!(target, "edlsrc has no uri");
            return Err(G2gError::NotConfigured);
        }
        self.timeline = timeline(&self.segments).await.map_err(|reason| {
            g2g_error!(target, "cannot play {}: {reason}", self.uri);
            G2gError::NotConfigured
        })?;
        self.play_from(0)?;
        let current = self.current.as_ref().expect("the timeline has a segment");
        match current.messages.recv().await {
            Some(SegmentMessage::Caps(caps)) => Ok(caps),
            Some(SegmentMessage::Failed(error)) => Err(error),
            Some(SegmentMessage::End) | None => {
                g2g_error!(
                    target,
                    "the first segment of {} decodes no frames",
                    self.uri
                );
                Err(G2gError::NotConfigured)
            }
            Some(SegmentMessage::Frame { .. }) => {
                unreachable!("SegmentSink sends the caps before the first frame")
            }
        }
    }
}

impl SourceLoop for EdlSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            if let Some(caps) = &self.caps {
                return Ok(caps.clone());
            }
            let caps = self.first_caps().await?;
            self.caps = Some(caps.clone());
            Ok(caps)
        })
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn query_duration(&self) -> Option<u64> {
        (!self.timeline.is_empty()).then(|| self.duration_ns())
    }

    fn seek_format(&self) -> Option<SeekFormat> {
        Some(SeekFormat::Time)
    }

    fn set_seek(&mut self, controller: SeekController) {
        self.seek = Some(controller);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut sequence = 0u64;
            loop {
                if let Some(seek) = self
                    .seek
                    .as_ref()
                    .and_then(|controller| controller.take_seek(flushing_only))
                {
                    out.push(PipelinePacket::Flush).await?;
                    let segment = Segment::for_flush_seek(&seek, Some(self.duration_ns()));
                    self.play_from(segment.start)?;
                    out.push(PipelinePacket::Segment(segment)).await?;
                    continue;
                }
                let Some(current) = &self.current else {
                    if !eos_then_wait_for_seek(self.seek.as_ref(), out).await? {
                        return Ok(sequence);
                    }
                    continue;
                };
                let segment = &self.timeline[current.timeline_index];
                match current.messages.recv().await {
                    Some(SegmentMessage::Caps(caps)) => {
                        if self.caps.as_ref() != Some(&caps) {
                            self.caps = Some(caps.clone());
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                        }
                    }
                    Some(SegmentMessage::Frame {
                        mut frame,
                        stream_time_ns,
                    }) => {
                        if stream_time_ns.saturating_add(CUT_TOLERANCE_NS) >= segment.end_ns() {
                            self.advance()?;
                            continue;
                        }
                        let pts_ns = segment
                            .timeline_start_ns
                            .saturating_add(stream_time_ns.saturating_sub(segment.start_ns));
                        frame.timing.pts_ns = pts_ns;
                        frame.timing.dts_ns = pts_ns;
                        frame.sequence = sequence;
                        sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    Some(SegmentMessage::End) => self.advance()?,
                    Some(SegmentMessage::Failed(error)) => return Err(error),
                    None => {
                        g2g_error!(
                            Target::category(short_type_name::<Self>()),
                            "the decoder thread of {} stopped without ending its segment",
                            segment.file
                        );
                        return Err(G2gError::Shutdown);
                    }
                }
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        EDLSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "EDL timeline source",
            "Source/Video",
            "Plays the segments of an mpv edl:// timeline back to back as one video stream",
            crate::AARON_BOXER,
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "uri" => {
                let uri = value.as_str().ok_or(PropError::Type)?;
                self.segments = single_stream(uri).map_err(|reason| {
                    g2g_error!(
                        Target::category(short_type_name::<Self>()),
                        "cannot play {uri}: {reason}"
                    );
                    PropError::Value
                })?;
                self.uri = uri.to_string();
                self.timeline.clear();
                self.caps = None;
                self.current = None;
                self.next = None;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "uri" => Some(PropValue::Str(self.uri.clone())),
            _ => None,
        }
    }
}

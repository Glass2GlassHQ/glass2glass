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
    attach_seek, bounded, eos_then_wait_for_seek, flushing_only, is_raw_audio, is_raw_video,
    run_graph, GraphNodeRef, Receiver, Registry, SeekController, SeekFormat, Sender, SourceLoop,
};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    Frame, G2gError, MemoryDomain, NodeKind, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, RawVideoFormat, Seek, Segment, SystemSlice,
};

use crate::clock::WallClock;
use crate::discover::discover;
use crate::edl::{parse_edl_uri, EdlSegment, EdlStream};
use crate::pcmclip::clip_pcm_to_segment;
use crate::uridecodebin::PLAYBIN_MAX_DEPTH;
use crate::videoconvert::VideoConvert;
use crate::videofx::any_geometry;

static EDLSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("uri", PropKind::Str, "edl:// timeline in mpv's EDL syntax"),
    PropertySpec::new(
        "stream",
        PropKind::Uint,
        "index of the timeline's stream to play, each !new_stream starts the next",
    )
    .with_default("0"),
    PropertySpec::new(
        "media",
        PropKind::Str,
        "what to decode from the stream's files, NV12 video or raw audio",
    )
    .with_enum_values("video | audio")
    .with_default("video"),
];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum EdlMedia {
    #[default]
    Video,
    Audio,
}

impl EdlMedia {
    fn parse(name: &str) -> Option<EdlMedia> {
        match name {
            "video" => Some(EdlMedia::Video),
            "audio" => Some(EdlMedia::Audio),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            EdlMedia::Video => "video",
            EdlMedia::Audio => "audio",
        }
    }

    fn is_decoded(self, caps: &Caps) -> bool {
        match self {
            EdlMedia::Video => is_raw_video(caps),
            EdlMedia::Audio => is_raw_audio(caps),
        }
    }

    fn sink_constraint(self) -> CapsConstraint<'static> {
        match self {
            EdlMedia::Video => {
                CapsConstraint::Accepts(CapsSet::one(any_geometry(RawVideoFormat::Nv12)))
            }
            EdlMedia::Audio => CapsConstraint::AcceptsAny,
        }
    }
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

#[derive(Debug)]
struct SegmentSink {
    media: EdlMedia,
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

    fn stream_time_ns(&self, pts_ns: u64) -> Option<u64> {
        match self.media {
            EdlMedia::Video => self
                .segment
                .presents(pts_ns)
                .then(|| self.segment.to_stream_time(pts_ns))
                .flatten(),
            // audio buffers straddle the seek target and edlsrc trims them to the timeline segment
            EdlMedia::Audio => u64::try_from(
                i128::from(self.segment.time) + i128::from(pts_ns) - i128::from(self.segment.start),
            )
            .ok(),
        }
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
        self.media.sink_constraint()
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
                    let Some(stream_time_ns) = self.stream_time_ns(frame.timing.pts_ns) else {
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
    media: EdlMedia,
    from_ns: u64,
    messages: Sender<SegmentMessage>,
) -> Result<(), G2gError> {
    const SEGMENT_LINK_CAPACITY: usize = 2;
    let target = Target::category(short_type_name::<EdlSrc>());
    let sink = SegmentSink {
        media,
        messages,
        segment: Segment::new(),
        caps: None,
        sent_caps: None,
    };
    let mut graph = crate::registry::default_registry()
        .build_uridecodebin(
            &format!("file://{file}"),
            sink,
            &|caps: &Caps| media.is_decoded(caps),
            PLAYBIN_MAX_DEPTH,
        )
        .map_err(|error| {
            g2g_error!(target, "cannot decode {file}: {error:?}");
            G2gError::NotConfigured
        })?;
    if media == EdlMedia::Video {
        // openh264 decodes to I420, and build_uridecodebin plugs no converter after its decoder
        let into_sink = graph
            .edges()
            .iter()
            .position(|edge| graph.node_kind(edge.dst.node) == Some(NodeKind::Sink))
            .expect("build_uridecodebin links its sink");
        graph.insert_on_edge(
            into_sink,
            GraphNodeRef::element(VideoConvert::new(RawVideoFormat::Nv12)),
        );
    }
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
        media: EdlMedia,
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
                    if let Err(error) = decode_segment(&file, media, from_ns, sender).await {
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

fn clip_audio_to_segment(
    frame_at_stream_time: Frame,
    segment: &TimelineSegment,
    caps: &Caps,
) -> Result<Option<Frame>, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let sample_bytes = format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?;
    let edges = Segment {
        start: segment.start_ns,
        stop: Some(segment.end_ns()),
        ..Segment::new()
    };
    let pcm = frame_at_stream_time
        .domain
        .require_system_bytes(short_type_name::<EdlSrc>())?
        .into_owned();
    let clipped = clip_pcm_to_segment(
        &edges,
        frame_at_stream_time.timing,
        pcm,
        sample_bytes * usize::from(*channels),
        *sample_rate,
    )?;
    Ok(clipped.map(|(timing, pcm)| {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            timing,
            frame_at_stream_time.sequence,
        )
    }))
}

#[derive(Debug, Default)]
pub struct EdlSrc {
    uri: String,
    streams: Vec<EdlStream>,
    stream_index: usize,
    media: EdlMedia,
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
        self.current = Some(SegmentPlayback::start(
            &self.timeline,
            index,
            offset_ns,
            self.media,
        )?);
        self.start_next(index)
    }

    fn start_next(&mut self, current_index: usize) -> Result<(), G2gError> {
        let next_index = current_index + 1;
        if next_index < self.timeline.len() {
            self.next = Some(SegmentPlayback::start(
                &self.timeline,
                next_index,
                0,
                self.media,
            )?);
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
        if self.streams.is_empty() {
            g2g_error!(target, "edlsrc has no uri");
            return Err(G2gError::NotConfigured);
        }
        let Some(stream) = self.streams.get(self.stream_index) else {
            g2g_error!(
                target,
                "{} has {} streams, there is no stream {}",
                self.uri,
                self.streams.len(),
                self.stream_index
            );
            return Err(G2gError::NotConfigured);
        };
        self.timeline = timeline(&stream.segments).await.map_err(|reason| {
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
                        let cut_tolerance_ns = match self.media {
                            EdlMedia::Video => CUT_TOLERANCE_NS,
                            EdlMedia::Audio => 0,
                        };
                        if stream_time_ns.saturating_add(cut_tolerance_ns) >= segment.end_ns() {
                            self.advance()?;
                            continue;
                        }
                        frame.timing.pts_ns = stream_time_ns;
                        if self.media == EdlMedia::Audio {
                            let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
                            let Some(clipped) = clip_audio_to_segment(frame, segment, caps)? else {
                                continue;
                            };
                            frame = clipped;
                        }
                        let pts_ns = segment
                            .timeline_start_ns
                            .saturating_add(frame.timing.pts_ns.saturating_sub(segment.start_ns));
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
            "Source/Video/Audio",
            "Plays the segments of one stream of an mpv edl:// timeline back to back",
            crate::AARON_BOXER,
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "uri" => {
                let uri = value.as_str().ok_or(PropError::Type)?;
                self.streams = parse_edl_uri(uri)
                    .map_err(|error| {
                        g2g_error!(
                            Target::category(short_type_name::<Self>()),
                            "cannot play {uri}: {error}"
                        );
                        PropError::Value
                    })?
                    .streams;
                self.uri = uri.to_string();
            }
            "stream" => {
                let index = value.as_uint().ok_or(PropError::Type)?;
                self.stream_index = usize::try_from(index).map_err(|_| PropError::Value)?;
            }
            "media" => {
                let media = value.as_str().ok_or(PropError::Type)?;
                self.media = EdlMedia::parse(media).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        self.timeline.clear();
        self.caps = None;
        self.current = None;
        self.next = None;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "uri" => Some(PropValue::Str(self.uri.clone())),
            "stream" => Some(PropValue::Uint(self.stream_index as u64)),
            "media" => Some(PropValue::Str(self.media.name().to_string())),
            _ => None,
        }
    }
}

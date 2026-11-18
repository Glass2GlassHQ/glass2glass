// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use g2g_core::caps::{framerate_q16_period_ns, framerate_q16_to_fraction};
use g2g_core::g2g_error;
use g2g_core::graph::{Graph, NodeId};
use g2g_core::log::Target;
use g2g_core::runtime::{
    attach_seek, attach_seek_controller, launch_quote, parse_chain, parse_launch, GraphMutator,
    GraphNode, GraphNodeRef, LaunchFactory, ParseError, PipelineProgress, Registry, RunBuilder,
    SeekController, SeekOutcome, SeekTicket, StateController,
};
use g2g_core::segment::Seek;
use g2g_core::state::PipelineState;
use g2g_core::{Bus, BusMessage};
use g2g_plugins::clock::WallClock;
use g2g_plugins::edl::{parse_edl_uri, EdlError};
use g2g_plugins::launched::RunFailure;
use g2g_plugins::registry::default_registry;

mod caps_probe;
mod demux_probe;
mod level_meter;
mod picture_filters;
mod render;
mod subtitles;
mod videosink;

pub use g2g_core::{ChannelLayout, ChannelPosition};
#[cfg(feature = "ffmpeg")]
pub use g2g_plugins::ffmpegdec::Lowres;
pub use g2g_plugins::glnv12::GlTarget;
pub use level_meter::AudioLevels;
pub use picture_filters::VideoFilterError;
pub use render::GlRenderContext;
#[cfg(feature = "wgpu")]
pub use render::WgpuRenderContext;
pub use subtitles::SubtitleSlot;

use caps_probe::CapsProbe;
use demux_probe::DemuxProbe;
use level_meter::{apply_level_meter, insert_level_meter, LEVEL_METER};
use picture_filters::{
    launch_chain, parse_filters, video_size_of, PictureFilter, VIDEO_PARAMS_PROBE,
};
use subtitles::Subtitles;
use videosink::{FrameSlot, PlayerVideoSink, PLAYER_VIDEO_SINK, PLAYER_VIDEO_SINK_METADATA};

// matches g2g-launch, low latency without starving the source
const LINK_CAPACITY: usize = 4;
const PLAYER_BUS_CAPACITY: usize = 256;
const PROPERTY_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy)]
pub enum Media<'a> {
    Uri(&'a str),
    LaunchLine(&'a str),
}

// mpv's per-file options of the same names, a pause of None keeps the current state
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    pub start: Option<Duration>,
    pub pause: Option<bool>,
}

// mpv's ao, where Discard is ao=null
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoundOutput {
    #[default]
    Device,
    Discard,
}

impl SoundOutput {
    fn sink(self) -> &'static str {
        match self {
            SoundOutput::Device => "autoaudiosink",
            SoundOutput::Discard => "fakeaudiosink",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioParams {
    pub sample_rate: u32,
    pub channel_count: u8,
    // unspecified when the stream names no speakers and the count has no conventional layout
    pub channel_layout: ChannelLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayRectangle {
    // pixels of the picture after the video filters
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
    pub colour: [u8; 3],
    // straight, not premultiplied
    pub alpha: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsdDimensions {
    pub width: u32,
    pub height: u32,
    pub margin_left: u32,
    pub margin_top: u32,
    pub margin_right: u32,
    pub margin_bottom: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Pause,
    TimePos,
    Duration,
    EofReached,
    // the decoder's picture size
    VideoParams,
    // the picture size after the video filters
    VideoOutParams,
    OsdDimensions,
    // the decoded audio, before it is converted for the device
    AudioParams,
    // frames per second of the video stream
    ContainerFps,
    // the picture size the container declares, before a decoder shrinks it, mpv's demux-w/h
    DemuxSize,
    // frames never shown, late past max-lateness or replaced before the host rendered them
    FrameDropCount,
    // frames shown after their deadline
    DelayedFrameCount,
    // frames per second over the last frame durations after the video filters
    EstimatedVfFps,
    // how far the demuxer has read past time-pos
    DemuxerCacheDuration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropertyChange {
    Pause(bool),
    TimePos(Option<Duration>),
    Duration(Option<Duration>),
    EofReached(bool),
    VideoParams(Option<VideoSize>),
    VideoOutParams(Option<VideoSize>),
    OsdDimensions(Option<OsdDimensions>),
    AudioParams(Option<AudioParams>),
    ContainerFps(Option<f64>),
    DemuxSize(Option<VideoSize>),
    FrameDropCount(u64),
    DelayedFrameCount(u64),
    EstimatedVfFps(Option<f64>),
    DemuxerCacheDuration(Option<Duration>),
}

impl PropertyChange {
    fn property(self) -> Property {
        match self {
            PropertyChange::Pause(_) => Property::Pause,
            PropertyChange::TimePos(_) => Property::TimePos,
            PropertyChange::Duration(_) => Property::Duration,
            PropertyChange::EofReached(_) => Property::EofReached,
            PropertyChange::DemuxSize(_) => Property::DemuxSize,
            PropertyChange::FrameDropCount(_) => Property::FrameDropCount,
            PropertyChange::DelayedFrameCount(_) => Property::DelayedFrameCount,
            PropertyChange::EstimatedVfFps(_) => Property::EstimatedVfFps,
            PropertyChange::DemuxerCacheDuration(_) => Property::DemuxerCacheDuration,
            PropertyChange::VideoParams(_) => Property::VideoParams,
            PropertyChange::VideoOutParams(_) => Property::VideoOutParams,
            PropertyChange::OsdDimensions(_) => Property::OsdDimensions,
            PropertyChange::AudioParams(_) => Property::AudioParams,
            PropertyChange::ContainerFps(_) => Property::ContainerFps,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndFile {
    Eof,
    Stop,
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    FileLoaded,
    EndFile(EndFile),
    Seek,
    PlaybackRestart,
    PropertyChange(PropertyChange),
    VideoReconfig,
}

#[derive(Debug)]
pub enum LoadError {
    Parse(ParseError),
    Edl(EdlError),
    NoVideoSink,
    StartNotSeekable,
    Spawn(std::io::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Parse(error) => write!(f, "parse error: {error}"),
            LoadError::Edl(error) => write!(f, "edl:// timeline: {error}"),
            LoadError::NoVideoSink => {
                write!(f, "the launch line has no {PLAYER_VIDEO_SINK} element")
            }
            LoadError::StartNotSeekable => {
                f.write_str("a start position was given, but the media cannot seek")
            }
            LoadError::Spawn(error) => write!(f, "failed to spawn the playback thread: {error}"),
        }
    }
}

impl std::error::Error for LoadError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekError {
    NoMedia,
    NotSeekable,
    NoPosition,
    UnknownFrameRate,
}

impl fmt::Display for SeekError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeekError::NoMedia => f.write_str("nothing is loaded"),
            SeekError::NotSeekable => f.write_str("the loaded media cannot seek"),
            SeekError::NoPosition => f.write_str("no frame has been shown yet"),
            SeekError::UnknownFrameRate => f.write_str("the video has no fixed frame rate"),
        }
    }
}

impl std::error::Error for SeekError {}

type WakeupCallback = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct EventQueue {
    pending: Mutex<VecDeque<Event>>,
    arrived: Condvar,
    wakeup: Mutex<Option<WakeupCallback>>,
}

impl fmt::Debug for EventQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventQueue").finish_non_exhaustive()
    }
}

impl EventQueue {
    fn push(&self, event: Event) {
        self.pending
            .lock()
            .expect("event queue lock")
            .push_back(event);
        self.arrived.notify_all();
        let wakeup = self.wakeup.lock().expect("wakeup callback lock").clone();
        if let Some(wakeup) = wakeup {
            wakeup();
        }
    }

    fn wait(&self, timeout: Duration) -> Option<Event> {
        let pending = self.pending.lock().expect("event queue lock");
        let (mut pending, _) = self
            .arrived
            .wait_timeout_while(pending, timeout, |pending| pending.is_empty())
            .expect("event queue lock");
        pending.pop_front()
    }
}

fn fps_of(framerate_q16: u32) -> f64 {
    let (numerator, denominator) = framerate_q16_to_fraction(framerate_q16);
    f64::from(numerator) / f64::from(denominator)
}

#[derive(Debug, Default)]
pub(crate) struct Shared {
    events: EventQueue,
    frames: FrameSlot,
    paused: AtomicBool,
    eof_reached: AtomicBool,
    framerate_q16: Mutex<Option<u32>>,
    demux_size: Mutex<Option<VideoSize>>,
    // stream time of the furthest frame that reached the video decoder
    demuxed_position_ns: Mutex<Option<u64>>,
    frame_drop_count: AtomicU64,
    delayed_frame_count: AtomicU64,
    estimated_vf_fps: Mutex<Option<f64>>,
    video_params: Mutex<Option<VideoSize>>,
    video_out_params: Mutex<Option<VideoSize>>,
    osd_dimensions: Mutex<Option<OsdDimensions>>,
    audio_params: Mutex<Option<AudioParams>>,
    audio_levels: Mutex<Option<AudioLevels>>,
    // what time-pos shows from a seek request until its first frame is presented, as mpv's does
    seek_target: Mutex<Option<Duration>>,
    // set by a seek request, taken by the flush that seek sends through playervideosink
    seek_flush_pending: AtomicBool,
    overlay: Mutex<Vec<OverlayRectangle>>,
    subtitles: Mutex<Subtitles>,
    // the last value reported for each observed property
    observed: Mutex<Vec<PropertyChange>>,
}

impl Shared {
    fn report(&self, value: PropertyChange) {
        let mut observed = self.observed.lock().expect("observed properties lock");
        let Some(last) = observed
            .iter_mut()
            .find(|last| last.property() == value.property())
        else {
            return;
        };
        if *last == value {
            return;
        }
        *last = value;
        drop(observed);
        self.events.push(Event::PropertyChange(value));
    }

    fn set_paused(&self, paused: bool, state: Option<&StateController>) {
        self.paused.store(paused, Ordering::SeqCst);
        if let Some(state) = state {
            if paused {
                state.set_state(PipelineState::Paused);
            } else if state.is_prerolled() {
                state.set_state(PipelineState::Playing);
            }
        }
        self.report(PropertyChange::Pause(paused));
    }

    fn set_eof_reached(&self, reached: bool) {
        self.eof_reached.store(reached, Ordering::SeqCst);
        self.report(PropertyChange::EofReached(reached));
    }

    fn reach_end(&self, state: &StateController) {
        self.set_paused(true, Some(state));
        self.set_eof_reached(true);
    }

    pub(crate) fn set_demux_size(&self, size: Option<VideoSize>) {
        *self.demux_size.lock().expect("demux size lock") = size;
        self.report(PropertyChange::DemuxSize(size));
    }

    pub(crate) fn set_demuxed_position(&self, stream_time_ns: Option<u64>) {
        *self
            .demuxed_position_ns
            .lock()
            .expect("demuxed position lock") = stream_time_ns;
    }

    fn demuxer_cache_duration(&self, progress: Option<&PipelineProgress>) -> Option<Duration> {
        let demuxed_ns = (*self
            .demuxed_position_ns
            .lock()
            .expect("demuxed position lock"))?;
        let played = self.time_pos(progress)?;
        Some(Duration::from_nanos(demuxed_ns).saturating_sub(played))
    }

    pub(crate) fn set_frame_drop_count(&self, count: u64) {
        self.frame_drop_count.store(count, Ordering::SeqCst);
        self.report(PropertyChange::FrameDropCount(count));
    }

    pub(crate) fn set_delayed_frame_count(&self, count: u64) {
        self.delayed_frame_count.store(count, Ordering::SeqCst);
        self.report(PropertyChange::DelayedFrameCount(count));
    }

    pub(crate) fn set_estimated_vf_fps(&self, fps: Option<f64>) {
        *self.estimated_vf_fps.lock().expect("estimated fps lock") = fps;
        self.report(PropertyChange::EstimatedVfFps(fps));
    }

    pub(crate) fn set_video_params(&self, size: Option<VideoSize>) {
        *self.video_params.lock().expect("video params lock") = size;
        self.report(PropertyChange::VideoParams(size));
    }

    pub(crate) fn set_framerate_q16(&self, framerate_q16: Option<u32>) {
        *self.framerate_q16.lock().expect("frame rate lock") = framerate_q16;
        self.report(PropertyChange::ContainerFps(framerate_q16.map(fps_of)));
    }

    pub(crate) fn set_video_out_params(&self, size: Option<VideoSize>) {
        *self.video_out_params.lock().expect("video out params lock") = size;
        self.report(PropertyChange::VideoOutParams(size));
    }

    pub(crate) fn set_osd_dimensions(&self, dimensions: OsdDimensions) {
        *self.osd_dimensions.lock().expect("osd dimensions lock") = Some(dimensions);
        self.report(PropertyChange::OsdDimensions(Some(dimensions)));
    }

    pub(crate) fn set_audio_params(&self, params: Option<AudioParams>) {
        *self.audio_params.lock().expect("audio params lock") = params;
        self.report(PropertyChange::AudioParams(params));
    }

    fn set_audio_levels(&self, levels: Option<AudioLevels>) {
        *self.audio_levels.lock().expect("audio levels lock") = levels;
    }

    fn time_pos(&self, progress: Option<&PipelineProgress>) -> Option<Duration> {
        let seek_target = *self.seek_target.lock().expect("seek target lock");
        seek_target.or_else(|| {
            progress
                .and_then(PipelineProgress::position)
                .map(Duration::from_nanos)
        })
    }

    pub(crate) fn seek_landed(&self) {
        *self.seek_target.lock().expect("seek target lock") = None;
        self.seek_flush_pending.store(false, Ordering::SeqCst);
    }

    pub(crate) fn take_seek_flush(&self) -> bool {
        self.seek_flush_pending.swap(false, Ordering::SeqCst)
    }

    fn report_progress(&self, progress: &PipelineProgress) {
        self.report(PropertyChange::TimePos(self.time_pos(Some(progress))));
        self.report(PropertyChange::DemuxerCacheDuration(
            self.demuxer_cache_duration(Some(progress)),
        ));
        self.report(PropertyChange::Duration(
            progress.duration().map(Duration::from_nanos),
        ));
    }
}

#[derive(Debug)]
struct Playback {
    state: StateController,
    progress: PipelineProgress,
    seek: Option<SeekController>,
    seeks: tokio::sync::mpsc::UnboundedSender<Duration>,
    stopping: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

#[derive(Debug)]
pub struct Player {
    registry: Registry,
    shared: Arc<Shared>,
    playback: Mutex<Option<Playback>>,
    // mpv keeps its vf chain across loads
    picture_filters: tokio::sync::watch::Sender<Vec<PictureFilter>>,
    level_meter_window: tokio::sync::watch::Sender<Option<Duration>>,
    sound_output: Mutex<SoundOutput>,
    #[cfg(feature = "ffmpeg")]
    lowres: Mutex<Lowres>,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

fn nanoseconds(position: Duration) -> u64 {
    u64::try_from(position.as_nanos()).unwrap_or(u64::MAX)
}

fn edl_launch_line(uri: &str, sound_output: SoundOutput) -> Result<String, LoadError> {
    // guikit's timelines carry their sound file in the stream after the first !new_stream
    const SOUND_STREAM: usize = 1;
    let edl = parse_edl_uri(uri).map_err(LoadError::Edl)?;
    let quoted = launch_quote(uri);
    let mut line = format!("edlsrc uri={quoted} ! {PLAYER_VIDEO_SINK}");
    if edl.streams.len() > SOUND_STREAM {
        line.push_str(&format!(
            "  edlsrc uri={quoted} stream={SOUND_STREAM} media=audio ! audioconvert ! audioresample ! {}",
            sound_output.sink()
        ));
    }
    Ok(line)
}

impl Player {
    pub fn new() -> Self {
        let mut registry = default_registry();
        registry.register_launch(LaunchFactory::of::<PlayerVideoSink>(
            PLAYER_VIDEO_SINK,
            || Box::new(PlayerVideoSink::new(Arc::default())),
        ));
        Player {
            registry,
            shared: Arc::default(),
            playback: Mutex::new(None),
            picture_filters: tokio::sync::watch::Sender::new(Vec::new()),
            level_meter_window: tokio::sync::watch::Sender::new(None),
            sound_output: Mutex::new(SoundOutput::default()),
            #[cfg(feature = "ffmpeg")]
            lowres: Mutex::new(Lowres::default()),
        }
    }

    pub fn load(&self, media: Media<'_>, options: LoadOptions) -> Result<(), LoadError> {
        self.stop();
        let sound_output = *self.sound_output.lock().expect("sound output lock");
        let line = match media {
            Media::Uri(uri) if uri.starts_with("edl://") => edl_launch_line(uri, sound_output)?,
            Media::Uri(uri) => format!(
                "playbin uri={} video-sink={PLAYER_VIDEO_SINK} audio-sink={}",
                launch_quote(uri),
                sound_output.sink()
            ),
            Media::LaunchLine(line) => line.to_string(),
        };
        let mut graph = parse_launch(&self.registry, &line).map_err(LoadError::Parse)?;
        #[cfg(feature = "ffmpeg")]
        set_lowres_on_decoders(&mut graph, *self.lowres.lock().expect("lowres lock"));
        let video_sinks = connect_video_sinks(&mut graph, &self.shared);
        if video_sinks.is_empty() && matches!(media, Media::LaunchLine(_)) {
            return Err(LoadError::NoVideoSink);
        }
        let (decoded_size, decoded) = tokio::sync::watch::channel(None);
        if let Some(&first_sink) = video_sinks.first() {
            insert_video_params_probe(&mut graph, first_sink, &self.shared, decoded_size);
            insert_demux_probe(&mut graph, first_sink, &self.shared);
        }
        let mut level_meter_window = self.level_meter_window.subscribe();
        let has_level_meter = insert_level_meter(
            &mut graph,
            *level_meter_window.borrow_and_update(),
            &self.shared,
        );
        let seek = match options.start {
            // a source that cannot seek still plays, seek() refuses it
            None => attach_seek_controller(&mut graph).ok(),
            Some(start) => {
                let (seek, _ticket) = attach_seek(&mut graph, Seek::flush_to(nanoseconds(start)))
                    .map_err(|_| LoadError::StartNotSeekable)?;
                Some(seek)
            }
        };
        if let Some(seek) = &seek {
            seek.keep_run_open();
        }
        self.shared.set_eof_reached(false);
        self.shared.set_framerate_q16(None);
        if let Some(pause) = options.pause {
            self.shared.set_paused(pause, None);
        }
        // audio and video start together only once every sink has prerolled
        let state = StateController::new(PipelineState::Paused);
        self.shared.seek_landed();
        let progress = PipelineProgress::new();
        let (seeks, seek_requests) = tokio::sync::mpsc::unbounded_channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let run = PlaybackRun {
            graph,
            state: state.clone(),
            progress: progress.clone(),
            seek: seek.clone(),
            seek_requests,
            stopping: stopping.clone(),
            shared: self.shared.clone(),
            picture_filters: self.picture_filters.subscribe(),
            decoded,
            level_meter_window: has_level_meter.then_some(level_meter_window),
            has_video_sink: !video_sinks.is_empty(),
        };
        let thread = std::thread::Builder::new()
            .name("g2g-player".into())
            .spawn(move || run.run())
            .map_err(LoadError::Spawn)?;
        *self.playback.lock().expect("playback lock") = Some(Playback {
            state,
            progress,
            seek,
            seeks,
            stopping,
            thread,
        });
        self.shared.events.push(Event::FileLoaded);
        Ok(())
    }

    pub fn stop(&self) {
        let Some(playback) = self.playback.lock().expect("playback lock").take() else {
            return;
        };
        playback.stopping.store(true, Ordering::SeqCst);
        // g2g-core ends a run held open at the end of the stream only on shutdown
        if let Some(seek) = &playback.seek {
            seek.shutdown();
        }
        playback.state.set_state(PipelineState::Null);
        playback
            .thread
            .join()
            .expect("the playback thread catches its own errors");
        self.shared.seek_landed();
        self.shared.set_demux_size(None);
        self.shared.set_demuxed_position(None);
        self.shared.set_frame_drop_count(0);
        self.shared.set_delayed_frame_count(0);
        self.shared.set_estimated_vf_fps(None);
        self.shared.set_video_params(None);
        self.shared.set_video_out_params(None);
        self.shared.set_framerate_q16(None);
        self.shared.set_audio_params(None);
        self.shared.set_audio_levels(None);
        // mpv drops the tracks of sub-add on every load
        self.shared
            .subtitles
            .lock()
            .expect("subtitles lock")
            .drop_files();
        self.shared.frames.request_redraw();
    }

    pub fn pause(&self) {
        self.set_paused(true);
    }

    pub fn play(&self) {
        self.set_paused(false);
    }

    fn set_paused(&self, paused: bool) {
        let playback = self.playback.lock().expect("playback lock");
        self.shared
            .set_paused(paused, playback.as_ref().map(|playback| &playback.state));
    }

    pub fn seek(&self, position: Duration) -> Result<(), SeekError> {
        self.seek_showing(position, position)
    }

    fn seek_showing(&self, target: Duration, shown: Duration) -> Result<(), SeekError> {
        let playback = self.playback.lock().expect("playback lock");
        let playback = playback.as_ref().ok_or(SeekError::NoMedia)?;
        if playback.seek.is_none() {
            return Err(SeekError::NotSeekable);
        }
        *self.shared.seek_target.lock().expect("seek target lock") = Some(shown);
        self.shared.seek_flush_pending.store(true, Ordering::SeqCst);
        // the playback thread is gone once the file ended, nothing is loaded then
        playback.seeks.send(target).map_err(|_| {
            self.shared.seek_landed();
            SeekError::NoMedia
        })
    }

    fn seek_from_position(
        &self,
        target_and_shown: impl FnOnce(Duration) -> (Duration, Duration),
    ) -> Result<(), SeekError> {
        let position = {
            let playback = self.playback.lock().expect("playback lock");
            let playback = playback.as_ref().ok_or(SeekError::NoMedia)?;
            self.shared
                .time_pos(Some(&playback.progress))
                .ok_or(SeekError::NoPosition)?
        };
        let (target, shown) = target_and_shown(position);
        self.seek_showing(target, shown)
    }

    pub fn seek_forward(&self, offset: Duration) -> Result<(), SeekError> {
        self.seek_from_position(|position| {
            let target = position.saturating_add(offset);
            (target, target)
        })
    }

    pub fn seek_backward(&self, offset: Duration) -> Result<(), SeekError> {
        self.seek_from_position(|position| {
            let target = position.saturating_sub(offset);
            (target, target)
        })
    }

    fn frame_period(&self) -> Result<Duration, SeekError> {
        let framerate_q16 = self
            .shared
            .framerate_q16
            .lock()
            .expect("frame rate lock")
            .ok_or(SeekError::UnknownFrameRate)?;
        let period =
            framerate_q16_period_ns(framerate_q16).expect("stored frame rates are never zero");
        Ok(Duration::from_nanos(period))
    }

    pub fn frame_step(&self) -> Result<(), SeekError> {
        let period = self.frame_period()?;
        self.pause();
        // seeks land on the first frame at or after the target
        self.seek_from_position(|position| {
            (
                position.saturating_add(period / 2),
                position.saturating_add(period),
            )
        })
    }

    pub fn frame_back_step(&self) -> Result<(), SeekError> {
        let period = self.frame_period()?;
        self.pause();
        self.seek_from_position(|position| {
            (
                position.saturating_sub(period + period / 2),
                position.saturating_sub(period),
            )
        })
    }

    pub fn property(&self, property: Property) -> PropertyChange {
        let playback = self.playback.lock().expect("playback lock");
        let progress = playback.as_ref().map(|playback| &playback.progress);
        match property {
            Property::Pause => PropertyChange::Pause(self.shared.paused.load(Ordering::SeqCst)),
            Property::TimePos => PropertyChange::TimePos(self.shared.time_pos(progress)),
            Property::Duration => PropertyChange::Duration(
                progress
                    .and_then(PipelineProgress::duration)
                    .map(Duration::from_nanos),
            ),
            Property::EofReached => {
                PropertyChange::EofReached(self.shared.eof_reached.load(Ordering::SeqCst))
            }
            Property::DemuxSize => {
                PropertyChange::DemuxSize(*self.shared.demux_size.lock().expect("demux size lock"))
            }
            Property::FrameDropCount => {
                PropertyChange::FrameDropCount(self.shared.frame_drop_count.load(Ordering::SeqCst))
            }
            Property::DelayedFrameCount => PropertyChange::DelayedFrameCount(
                self.shared.delayed_frame_count.load(Ordering::SeqCst),
            ),
            Property::EstimatedVfFps => PropertyChange::EstimatedVfFps(
                *self
                    .shared
                    .estimated_vf_fps
                    .lock()
                    .expect("estimated fps lock"),
            ),
            Property::DemuxerCacheDuration => {
                PropertyChange::DemuxerCacheDuration(self.shared.demuxer_cache_duration(progress))
            }
            Property::VideoParams => PropertyChange::VideoParams(
                *self.shared.video_params.lock().expect("video params lock"),
            ),
            Property::VideoOutParams => PropertyChange::VideoOutParams(
                *self
                    .shared
                    .video_out_params
                    .lock()
                    .expect("video out params lock"),
            ),
            Property::OsdDimensions => PropertyChange::OsdDimensions(
                *self
                    .shared
                    .osd_dimensions
                    .lock()
                    .expect("osd dimensions lock"),
            ),
            Property::AudioParams => PropertyChange::AudioParams(
                *self.shared.audio_params.lock().expect("audio params lock"),
            ),
            Property::ContainerFps => PropertyChange::ContainerFps(
                self.shared
                    .framerate_q16
                    .lock()
                    .expect("frame rate lock")
                    .map(fps_of),
            ),
        }
    }

    pub fn observe(&self, property: Property) {
        let current = self.property(property);
        let mut observed = self
            .shared
            .observed
            .lock()
            .expect("observed properties lock");
        if observed.iter().any(|last| last.property() == property) {
            return;
        }
        observed.push(current);
        drop(observed);
        self.shared.events.push(Event::PropertyChange(current));
    }

    pub fn wait_event(&self, timeout: Duration) -> Option<Event> {
        self.shared.events.wait(timeout)
    }

    pub fn set_update_callback(&self, update: impl Fn() + Send + Sync + 'static) {
        self.shared
            .frames
            .set_update_callback(Some(Arc::new(update)));
    }

    pub fn wants_redraw(&self) -> bool {
        self.shared.frames.wants_redraw()
    }

    pub fn set_overlay(&self, rectangles: Vec<OverlayRectangle>) {
        *self.shared.overlay.lock().expect("overlay lock") = rectangles;
        self.shared.frames.request_redraw();
    }

    pub fn set_subtitle_file(
        &self,
        slot: SubtitleSlot,
        path: Option<&std::path::Path>,
    ) -> std::io::Result<()> {
        let mut subtitles = self.shared.subtitles.lock().expect("subtitles lock");
        subtitles.load(slot, path)?;
        drop(subtitles);
        self.shared.frames.request_redraw();
        Ok(())
    }

    pub fn set_subtitle_visibility(&self, slot: SubtitleSlot, visible: bool) {
        let mut subtitles = self.shared.subtitles.lock().expect("subtitles lock");
        subtitles.set_visible(slot, visible);
        drop(subtitles);
        self.shared.frames.request_redraw();
    }

    pub fn set_video_filters(&self, filters: &[&str]) -> Result<(), VideoFilterError> {
        self.picture_filters.send_replace(parse_filters(filters)?);
        Ok(())
    }

    // takes effect at the next load
    pub fn set_sound_output(&self, output: SoundOutput) {
        *self.sound_output.lock().expect("sound output lock") = output;
    }

    // mpv's vd-lavc-o=lowres, takes effect at the next load
    #[cfg(feature = "ffmpeg")]
    pub fn set_lowres(&self, lowres: Lowres) {
        *self.lowres.lock().expect("lowres lock") = lowres;
    }

    pub fn set_level_meter(&self, window: Option<Duration>) {
        self.level_meter_window.send_replace(window);
    }

    pub fn audio_levels(&self) -> Option<AudioLevels> {
        self.shared
            .audio_levels
            .lock()
            .expect("audio levels lock")
            .clone()
    }

    pub fn set_wakeup_callback(&self, wakeup: impl Fn() + Send + Sync + 'static) {
        *self
            .shared
            .events
            .wakeup
            .lock()
            .expect("wakeup callback lock") = Some(Arc::new(wakeup));
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop();
    }
}

fn connect_video_sinks(graph: &mut Graph<GraphNode>, shared: &Arc<Shared>) -> Vec<NodeId> {
    let mut connected = Vec::new();
    for index in 0..graph.node_count() {
        let node = g2g_core::graph::NodeId(index as u32);
        let Some(element) = graph.element_mut(node) else {
            continue;
        };
        let is_player_sink = matches!(
            element,
            GraphNodeRef::Element(sink) if sink.metadata() == PLAYER_VIDEO_SINK_METADATA
        );
        if is_player_sink {
            // parse_launch builds each sink from a factory that cannot reach this player
            *element = GraphNodeRef::Element(Box::new(PlayerVideoSink::new(shared.clone())));
            connected.push(node);
        }
    }
    connected
}

fn insert_video_params_probe(
    graph: &mut Graph<GraphNode>,
    sink: NodeId,
    shared: &Arc<Shared>,
    decoded_size: tokio::sync::watch::Sender<Option<VideoSize>>,
) {
    let into_sink = graph
        .edges()
        .iter()
        .position(|edge| edge.dst.node == sink)
        .expect("parse_launch links every sink");
    let shared = shared.clone();
    let probe = CapsProbe::new(
        g2g_core::ElementMetadata::new(
            "Player video params probe",
            "Filter/Video",
            "Reports the decoded picture size to the g2g-player run",
            "Aaron Boxer <boxerab@protonmail.com>",
        ),
        move |caps| {
            let Some(size) = video_size_of(caps) else {
                return;
            };
            shared.set_video_params(Some(size));
            decoded_size.send_if_modified(|last| {
                let changed = *last != Some(size);
                *last = Some(size);
                changed
            });
        },
    );
    let probe = graph.insert_on_edge(into_sink, GraphNodeRef::element(probe));
    graph.set_node_name(probe, VIDEO_PARAMS_PROBE.to_string());
}

#[cfg(feature = "ffmpeg")]
fn set_lowres_on_decoders(graph: &mut Graph<GraphNode>, lowres: Lowres) {
    const LOWRES_PROPERTY: &str = "lowres";
    for index in 0..graph.node_count() {
        let Some(GraphNodeRef::Element(element)) = graph.element_mut(NodeId(index as u32)) else {
            continue;
        };
        if element
            .properties()
            .iter()
            .any(|spec| spec.name == LOWRES_PROPERTY)
        {
            element
                .set_property(
                    LOWRES_PROPERTY,
                    g2g_core::PropValue::Str(lowres.nick().into()),
                )
                .expect("ffmpegdec takes every Lowres nick");
        }
    }
}

fn is_video_decoder(graph: &Graph<GraphNode>, node: NodeId) -> bool {
    const VIDEO_DECODER_CLASSIFICATION: &str = "Codec/Decoder/Video";
    matches!(
        graph.element(node),
        Some(GraphNodeRef::Element(element))
            if element.metadata().classification.starts_with(VIDEO_DECODER_CLASSIFICATION)
    )
}

fn insert_demux_probe(graph: &mut Graph<GraphNode>, sink: NodeId, shared: &Arc<Shared>) {
    let edge_into = |graph: &Graph<GraphNode>, node: NodeId| {
        graph.edges().iter().position(|edge| edge.dst.node == node)
    };
    let mut node = sink;
    let into_decoder = loop {
        let Some(edge) = edge_into(graph, node) else {
            return;
        };
        node = graph.edges()[edge].src.node;
        if is_video_decoder(graph, node) {
            break edge_into(graph, node).expect("decoders have an input");
        }
    };
    graph.insert_on_edge(
        into_decoder,
        GraphNodeRef::element(DemuxProbe::new(shared.clone())),
    );
}

struct PlaybackRun {
    graph: Graph<GraphNode>,
    state: StateController,
    progress: PipelineProgress,
    seek: Option<SeekController>,
    seek_requests: tokio::sync::mpsc::UnboundedReceiver<Duration>,
    stopping: Arc<AtomicBool>,
    shared: Arc<Shared>,
    picture_filters: tokio::sync::watch::Receiver<Vec<PictureFilter>>,
    decoded: tokio::sync::watch::Receiver<Option<VideoSize>>,
    // None when the graph has no audio sink to meter
    level_meter_window: Option<tokio::sync::watch::Receiver<Option<Duration>>>,
    has_video_sink: bool,
}

async fn install_picture_filters(
    mutator: &GraphMutator<'_>,
    registry: &Registry,
    installed: Vec<String>,
    filters: Vec<PictureFilter>,
    decoded: Option<VideoSize>,
) -> Vec<String> {
    let target = Target::category("g2g-player");
    let mut remaining = installed;
    // GraphMutator::remove refuses an element whose removal hands the sink caps it does not accept
    while let Some(name) = remaining.pop() {
        if let Err(error) = mutator.remove(&name).await {
            g2g_error!(target, "cannot remove picture filter {name}: {error:?}");
            remaining.push(name);
            return remaining;
        }
    }
    let Some(decoded) = decoded.filter(|_| !filters.is_empty()) else {
        return Vec::new();
    };
    let elements = launch_chain(&filters, decoded).and_then(|chain| {
        parse_chain(registry, &chain)
            .map_err(|error| VideoFilterError::Malformed(error.to_string()))
    });
    let elements = match elements {
        Ok(elements) => elements,
        Err(error) => {
            g2g_error!(target, "{error}");
            return Vec::new();
        }
    };
    match mutator
        .insert_chain_after(VIDEO_PARAMS_PROBE, elements)
        .await
    {
        Ok(names) => names,
        Err(error) => {
            g2g_error!(target, "cannot insert the picture filters: {error:?}");
            Vec::new()
        }
    }
}

impl PlaybackRun {
    fn run(self) {
        let PlaybackRun {
            graph,
            state,
            progress,
            seek,
            mut seek_requests,
            stopping,
            shared,
            mut picture_filters,
            mut decoded,
            mut level_meter_window,
            has_video_sink,
        } = self;
        let registry = default_registry();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("build tokio runtime");
        let outcome = runtime.block_on(async {
            let clock = WallClock::new();
            let (bus, bus_handle) = Bus::new(PLAYER_BUS_CAPACITY);
            let mut builder = RunBuilder::new(graph, &clock, LINK_CAPACITY)
                .with_state(&state)
                .with_progress(&progress)
                .with_bus(&bus_handle);
            if let Some(seek) = &seek {
                builder = builder.with_seek_controller(seek);
            }
            let (mutator, run) = builder.run_mutable();
            let mutator = mutator.with_converters(g2g_plugins::registry::converters());
            tokio::pin!(run);
            let mut poll = tokio::time::interval(PROPERTY_POLL_INTERVAL);
            let mut pending_seek: Option<SeekTicket> = None;
            let mut installed_filters: Vec<String> = Vec::new();
            let mut pending_filters: Option<Pin<Box<dyn Future<Output = Vec<String>> + '_>>> = None;
            picture_filters.mark_changed();
            let mut filters_changed = false;
            let mut pending_level_meter: Option<Pin<Box<dyn Future<Output = ()> + '_>>> = None;
            let mut level_meter_changed = false;
            loop {
                if level_meter_changed && pending_level_meter.is_none() {
                    level_meter_changed = false;
                    let window = *level_meter_window
                        .as_mut()
                        .expect("set only with a meter")
                        .borrow_and_update();
                    if window.is_none() {
                        shared.set_audio_levels(None);
                    }
                    pending_level_meter = Some(Box::pin(apply_level_meter(&mutator, window)));
                }
                if filters_changed && pending_filters.is_none() {
                    filters_changed = false;
                    pending_filters = Some(Box::pin(install_picture_filters(
                        &mutator,
                        &registry,
                        core::mem::take(&mut installed_filters),
                        picture_filters.borrow_and_update().clone(),
                        *decoded.borrow_and_update(),
                    )));
                }
                tokio::select! {
                    outcome = &mut run => break outcome,
                    Ok(()) = async { level_meter_window.as_mut().expect("guarded by is_some").changed().await }, if level_meter_window.is_some() => {
                        level_meter_changed = true;
                    }
                    () = async { pending_level_meter.as_mut().expect("guarded by is_some").await }, if pending_level_meter.is_some() => {
                        pending_level_meter = None;
                    }
                    Some(message) = bus.recv() => {
                        if let BusMessage::Level { element, peak_db, rms_db, .. } = message {
                            // readings posted before the meter was switched off still arrive after it
                            let metering = level_meter_window
                                .as_ref()
                                .is_some_and(|window| window.borrow().is_some());
                            if element == LEVEL_METER && metering {
                                shared.set_audio_levels(Some(AudioLevels { peak_db, rms_db }));
                            }
                        }
                    }
                    Ok(()) = picture_filters.changed() => filters_changed = true,
                    Ok(()) = decoded.changed() => filters_changed = true,
                    installed = async { pending_filters.as_mut().expect("guarded by is_some").await }, if pending_filters.is_some() => {
                        installed_filters = installed;
                        pending_filters = None;
                    }
                    Some(position) = seek_requests.recv() => {
                        let controller = seek.as_ref().expect("seek() refuses media that cannot seek");
                        // gstreamer's flushing seek prerolls every sink again before playing on
                        state.set_state(PipelineState::Paused);
                        state.request_repreroll();
                        pending_seek = Some(controller.seek(Seek::flush_to(nanoseconds(position))));
                        shared.set_eof_reached(false);
                        shared.events.push(Event::Seek);
                    }
                    outcome = async { pending_seek.as_ref().expect("guarded by is_some").wait().await }, if pending_seek.is_some() => {
                        pending_seek = None;
                        if outcome == SeekOutcome::Served {
                            shared.events.push(Event::PlaybackRestart);
                        }
                        // playervideosink ends the seek's time-pos with its first frame, nothing does without one
                        if !has_video_sink || outcome != SeekOutcome::Served {
                            shared.seek_landed();
                        }
                    }
                    () = state.await_prerolled(), if state.state() == PipelineState::Paused && !shared.paused.load(Ordering::SeqCst) => {
                        if !shared.paused.load(Ordering::SeqCst) {
                            state.set_state(PipelineState::Playing);
                        }
                    }
                    _ = poll.tick() => {
                        shared.report_progress(&progress);
                        let parked_at_end = seek.as_ref().is_some_and(SeekController::stream_ended);
                        if parked_at_end && !shared.eof_reached.load(Ordering::SeqCst) {
                            shared.reach_end(&state);
                        }
                    }
                }
            }
        });
        shared.report_progress(&progress);
        let end = match outcome {
            Ok(_) if stopping.load(Ordering::SeqCst) => EndFile::Stop,
            // a run with a muxer or a source that cannot seek ends at the end of the stream
            Ok(_) => {
                shared.reach_end(&state);
                EndFile::Eof
            }
            Err(error) => EndFile::Error(RunFailure::Error(error).to_string()),
        };
        shared.events.push(Event::EndFile(end));
    }
}

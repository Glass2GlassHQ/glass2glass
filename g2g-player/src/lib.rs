// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use g2g_core::graph::Graph;
use g2g_core::runtime::{
    attach_seek, attach_seek_controller, parse_launch, GraphNode, GraphNodeRef, LaunchFactory,
    ParseError, PipelineProgress, Registry, RunBuilder, SeekController, SeekOutcome, SeekTicket,
    StateController,
};
use g2g_core::segment::Seek;
use g2g_core::state::PipelineState;
use g2g_plugins::clock::WallClock;
use g2g_plugins::launched::RunFailure;
use g2g_plugins::registry::default_registry;

mod render;
mod videosink;

pub use g2g_plugins::glnv12::GlTarget;
pub use render::GlRenderContext;
#[cfg(feature = "wgpu")]
pub use render::WgpuRenderContext;

use videosink::{FrameSlot, PlayerVideoSink, PLAYER_VIDEO_SINK, PLAYER_VIDEO_SINK_METADATA};

// matches g2g-launch, low latency without starving the source
const LINK_CAPACITY: usize = 4;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Pause,
    TimePos,
    Duration,
    EofReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyChange {
    Pause(bool),
    TimePos(Option<Duration>),
    Duration(Option<Duration>),
    EofReached(bool),
}

impl PropertyChange {
    fn property(self) -> Property {
        match self {
            PropertyChange::Pause(_) => Property::Pause,
            PropertyChange::TimePos(_) => Property::TimePos,
            PropertyChange::Duration(_) => Property::Duration,
            PropertyChange::EofReached(_) => Property::EofReached,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndFile {
    Eof,
    Stop,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    NoVideoSink,
    StartNotSeekable,
    Spawn(std::io::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Parse(error) => write!(f, "parse error: {error}"),
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
}

impl fmt::Display for SeekError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeekError::NoMedia => f.write_str("nothing is loaded"),
            SeekError::NotSeekable => f.write_str("the loaded media cannot seek"),
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

#[derive(Debug, Default)]
pub(crate) struct Shared {
    events: EventQueue,
    frames: FrameSlot,
    paused: AtomicBool,
    eof_reached: AtomicBool,
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
            state.set_state(if paused {
                PipelineState::Paused
            } else {
                PipelineState::Playing
            });
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

    fn report_progress(&self, progress: &PipelineProgress) {
        self.report(PropertyChange::TimePos(
            progress.position().map(Duration::from_nanos),
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
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

fn nanoseconds(position: Duration) -> u64 {
    u64::try_from(position.as_nanos()).unwrap_or(u64::MAX)
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
        }
    }

    pub fn load(&self, media: Media<'_>, options: LoadOptions) -> Result<(), LoadError> {
        self.stop();
        let line = match media {
            Media::Uri(uri) => format!("playbin uri=\"{uri}\" video-sink={PLAYER_VIDEO_SINK}"),
            Media::LaunchLine(line) => line.to_string(),
        };
        let mut graph = parse_launch(&self.registry, &line).map_err(LoadError::Parse)?;
        let video_sinks = connect_video_sinks(&mut graph, &self.shared);
        if video_sinks == 0 && matches!(media, Media::LaunchLine(_)) {
            return Err(LoadError::NoVideoSink);
        }
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
        if let Some(pause) = options.pause {
            self.shared.set_paused(pause, None);
        }
        let state = StateController::new(if self.shared.paused.load(Ordering::SeqCst) {
            PipelineState::Paused
        } else {
            PipelineState::Playing
        });
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
        let playback = self.playback.lock().expect("playback lock");
        let playback = playback.as_ref().ok_or(SeekError::NoMedia)?;
        if playback.seek.is_none() {
            return Err(SeekError::NotSeekable);
        }
        // the playback thread is gone once the file ended, nothing is loaded then
        playback
            .seeks
            .send(position)
            .map_err(|_| SeekError::NoMedia)
    }

    pub fn property(&self, property: Property) -> PropertyChange {
        let playback = self.playback.lock().expect("playback lock");
        let progress = playback.as_ref().map(|playback| &playback.progress);
        match property {
            Property::Pause => PropertyChange::Pause(self.shared.paused.load(Ordering::SeqCst)),
            Property::TimePos => PropertyChange::TimePos(
                progress
                    .and_then(PipelineProgress::position)
                    .map(Duration::from_nanos),
            ),
            Property::Duration => PropertyChange::Duration(
                progress
                    .and_then(PipelineProgress::duration)
                    .map(Duration::from_nanos),
            ),
            Property::EofReached => {
                PropertyChange::EofReached(self.shared.eof_reached.load(Ordering::SeqCst))
            }
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

    pub fn has_new_frame(&self) -> bool {
        self.shared.frames.has_frame()
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

fn connect_video_sinks(graph: &mut Graph<GraphNode>, shared: &Arc<Shared>) -> usize {
    let mut connected = 0;
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
            connected += 1;
        }
    }
    connected
}

struct PlaybackRun {
    graph: Graph<GraphNode>,
    state: StateController,
    progress: PipelineProgress,
    seek: Option<SeekController>,
    seek_requests: tokio::sync::mpsc::UnboundedReceiver<Duration>,
    stopping: Arc<AtomicBool>,
    shared: Arc<Shared>,
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
        } = self;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("build tokio runtime");
        let outcome = runtime.block_on(async {
            let clock = WallClock::new();
            let mut builder =
                RunBuilder::new(graph, &clock, LINK_CAPACITY).with_state(&state).with_progress(&progress);
            if let Some(seek) = &seek {
                builder = builder.with_seek_controller(seek);
            }
            let run = builder.run();
            tokio::pin!(run);
            let mut poll = tokio::time::interval(PROPERTY_POLL_INTERVAL);
            let mut pending_seek: Option<SeekTicket> = None;
            loop {
                tokio::select! {
                    outcome = &mut run => break outcome,
                    Some(position) = seek_requests.recv() => {
                        let controller = seek.as_ref().expect("seek() refuses media that cannot seek");
                        if state.state() == PipelineState::Paused {
                            state.request_repreroll();
                        }
                        pending_seek = Some(controller.seek(Seek::flush_to(nanoseconds(position))));
                        shared.set_eof_reached(false);
                        shared.events.push(Event::Seek);
                    }
                    outcome = async { pending_seek.as_ref().expect("guarded by is_some").wait().await }, if pending_seek.is_some() => {
                        pending_seek = None;
                        if outcome == SeekOutcome::Served {
                            shared.events.push(Event::PlaybackRestart);
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

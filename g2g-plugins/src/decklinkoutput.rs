use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use alloc::boxed::Box;

use g2g_core::{
    g2g_error, ClockSync, DriftClock, FrameTiming, G2gError, HardwareError, MonotonicClock,
    PresentationPacer, RawVideoFormat,
};

use crate::decklink::{
    log_target, DeckLinkOpener, DeckLinkOutput, DisplayMode, ScheduledVideoFrame,
};

// slack between the first buffer and its display time
const PREROLL_NS: u64 = 200_000_000;

// GStreamer's decklinkvideosink hands a frame over 30 ms before it is due
const SCHEDULE_LEAD_NS: u64 = 30_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PictureLayout {
    pub(crate) mode: &'static DisplayMode,
    pub(crate) format: RawVideoFormat,
    pub(crate) row_bytes: u32,
    pub(crate) frame_bytes: usize,
}

#[derive(Default)]
struct OutputState {
    board: Option<Box<dyn DeckLinkOutput>>,
    video_enabled: bool,
    playing: bool,
    // elected-clock time that running time zero is due at
    anchor_ns: Option<i128>,
    // the first reference and board readings, so the drift clock starts out at the reference clock's time
    clock_epoch: Option<(u64, u64)>,
}

// the device a decklinkvideosink claims through its device-number
pub(crate) struct SharedOutput {
    device_number: u32,
    opener: DeckLinkOpener,
    clock: Arc<DriftClock>,
    state: Mutex<OutputState>,
}

impl core::fmt::Debug for SharedOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedOutput")
            .field("device_number", &self.device_number)
            .finish()
    }
}

fn shared_outputs() -> MutexGuard<'static, BTreeMap<u32, Weak<SharedOutput>>> {
    static OUTPUTS: Mutex<BTreeMap<u32, Weak<SharedOutput>>> = Mutex::new(BTreeMap::new());
    // the map holds only weak handles, so a panic elsewhere cannot leave it half written
    OUTPUTS.lock().unwrap_or_else(PoisonError::into_inner)
}

// a video sink's claim on its device, created on first claim
pub(crate) fn claim_output(device_number: u32, opener: DeckLinkOpener) -> Arc<SharedOutput> {
    let mut outputs = shared_outputs();
    if let Some(existing) = outputs.get(&device_number).and_then(Weak::upgrade) {
        return existing;
    }
    let output = Arc::new(SharedOutput {
        device_number,
        opener,
        clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
        state: Mutex::new(OutputState::default()),
    });
    outputs.insert(device_number, Arc::downgrade(&output));
    outputs.retain(|_, weak| weak.strong_count() > 0);
    output
}

fn signed(ns: u64) -> i128 {
    i128::from(ns)
}

fn unsigned(ns: i128) -> u64 {
    u64::try_from(ns.max(0)).unwrap_or(u64::MAX)
}

pub(crate) async fn sleep_ns(ns: u64) {
    tokio::time::sleep(Duration::from_nanos(ns)).await;
}

// waits until a buffer due at `deadline_ns` on the elected clock is close enough to hand to the board
pub(crate) async fn hold_until_due(sync: &ClockSync, deadline_ns: u64) {
    let wake = deadline_ns.saturating_sub(SCHEDULE_LEAD_NS);
    let now = sync.now_ns();
    if wake > now {
        sleep_ns(wake - now).await;
    }
}

impl SharedOutput {
    pub(crate) fn clock(&self) -> Arc<DriftClock> {
        Arc::clone(&self.clock)
    }

    fn lock(&self) -> Result<MutexGuard<'_, OutputState>, G2gError> {
        self.state
            .lock()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))
    }

    fn board<'a>(
        &self,
        state: &'a mut OutputState,
    ) -> Result<&'a mut Box<dyn DeckLinkOutput>, G2gError> {
        let board = match state.board.take() {
            Some(board) => board,
            None => (self.opener)(self.device_number)?,
        };
        Ok(state.board.insert(board))
    }

    // reads the board clock and folds the reading into the drift clock
    fn board_now(&self, state: &mut OutputState) -> Result<u64, G2gError> {
        let local_ns = self.clock.reference_now();
        let board_ns = self.board(state)?.hardware_time_ns()?;
        let (local_epoch, board_epoch) = *state.clock_epoch.get_or_insert((local_ns, board_ns));
        self.clock
            .observe(local_ns, local_epoch + board_ns.saturating_sub(board_epoch));
        Ok(board_ns)
    }

    pub(crate) fn enable_video(&self, mode: &'static DisplayMode) -> Result<(), G2gError> {
        let mut state = self.lock()?;
        if state.video_enabled {
            g2g_error!(
                log_target(),
                "device-number {} already has a video sink",
                self.device_number
            );
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.board(&mut state)?.enable_video(mode)?;
        state.video_enabled = true;
        Ok(())
    }

    // elected-clock time a buffer is due at, None for one outside the segment
    pub(crate) fn deadline_ns(
        &self,
        sync: &ClockSync,
        pacer: &PresentationPacer,
        pts_ns: u64,
    ) -> Result<Option<u64>, G2gError> {
        // an untimed buffer goes to the board at once
        if pts_ns == FrameTiming::PTS_NONE {
            return Ok(Some(sync.now_ns() + SCHEDULE_LEAD_NS));
        }
        let Some(running_time) = pacer.running_time(pts_ns) else {
            return Ok(None);
        };
        let mut state = self.lock()?;
        // a live path is due at the pipeline base time, as every other synchronising sink does
        let anchor = if sync.path_live() {
            signed(sync.base_time())
        } else {
            *state.anchor_ns.get_or_insert_with(|| {
                signed(sync.now_ns()) + signed(PREROLL_NS) - signed(running_time)
            })
        };
        let deadline = anchor + signed(running_time) + signed(sync.path_latency_min_ns());
        Ok(Some(unsigned(deadline)))
    }

    // board time of an elected-clock deadline, read against the board clock now
    fn stream_time_ns(
        &self,
        state: &mut OutputState,
        sync: &ClockSync,
        deadline_ns: u64,
    ) -> Result<(u64, u64), G2gError> {
        let board_now = self.board_now(state)?;
        let stream_time = signed(board_now) + signed(deadline_ns) - signed(sync.now_ns());
        Ok((board_now, unsigned(stream_time)))
    }

    // schedules one picture and returns the board time its display ends at
    pub(crate) fn schedule_video(
        &self,
        sync: &ClockSync,
        deadline_ns: u64,
        layout: &PictureLayout,
        pixels: &[u8],
    ) -> Result<u64, G2gError> {
        let mut state = self.lock()?;
        let (board_now, stream_time) = self.stream_time_ns(&mut state, sync, deadline_ns)?;
        let duration_ns = layout.mode.frame_duration_ns();
        // the board shows whole frames, so a picture waits for the next frame boundary
        let display_time_ns = stream_time
            .div_ceil(duration_ns)
            .checked_mul(duration_ns)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let playing = state.playing;
        let board = self.board(&mut state)?;
        board.schedule_video(&ScheduledVideoFrame {
            pixels,
            format: layout.format,
            width: layout.mode.width,
            height: layout.mode.height,
            row_bytes: layout.row_bytes,
            display_time_ns,
            duration_ns,
        })?;
        if !playing {
            // stream time then runs on from the board clock reading
            board.start_playback(board_now)?;
            state.playing = true;
        }
        Ok(display_time_ns + duration_ns)
    }

    // stops playback once the board clock reaches `stop_time_ns`, and returns the board time now
    pub(crate) fn stop_at(&self, stop_time_ns: u64) -> Result<u64, G2gError> {
        let mut state = self.lock()?;
        let board_now = self.board_now(&mut state)?;
        if state.playing {
            self.board(&mut state)?.stop_playback(stop_time_ns)?;
        }
        state.playing = false;
        state.anchor_ns = None;
        Ok(board_now)
    }

    pub(crate) fn flush(&self) -> Result<(), G2gError> {
        let mut state = self.lock()?;
        if state.playing {
            let board_now = self.board_now(&mut state)?;
            self.board(&mut state)?.stop_playback(board_now)?;
        }
        state.playing = false;
        state.anchor_ns = None;
        Ok(())
    }
}

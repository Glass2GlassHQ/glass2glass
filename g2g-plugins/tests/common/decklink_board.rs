#![allow(dead_code)] // not every test file uses every helper here

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, ChannelLayout, Colorimetry, Dim, FrameTiming, G2gError, Interlace,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::decklink::{DeckLinkOpener, DeckLinkOutput, DisplayMode, ScheduledVideoFrame};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledPicture {
    pub(crate) display_time_ns: u64,
    pub(crate) duration_ns: u64,
    pub(crate) row_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledSamples {
    pub(crate) stream_time_ns: u64,
    pub(crate) sample_frames: u32,
}

#[derive(Debug, Default)]
pub(crate) struct BoardLog {
    pub(crate) video_mode: Option<&'static str>,
    pub(crate) audio: Option<(AudioFormat, u8)>,
    pub(crate) pictures: Vec<ScheduledPicture>,
    pub(crate) samples: Vec<ScheduledSamples>,
    pub(crate) audio_prerolls: usize,
    pub(crate) playback_starts: Vec<u64>,
    pub(crate) playback_stops: Vec<u64>,
    // most sample frames one call takes, None takes all
    pub(crate) audio_frames_per_call: Option<u32>,
}

pub(crate) type SharedLog = Arc<Mutex<BoardLog>>;

pub(crate) type BoardTime = Arc<dyn Fn() -> u64 + Send + Sync>;

struct MockBoard {
    log: SharedLog,
    time: BoardTime,
}

impl MockBoard {
    fn log(&self) -> std::sync::MutexGuard<'_, BoardLog> {
        self.log.lock().expect("board log lock")
    }
}

impl DeckLinkOutput for MockBoard {
    fn enable_video(&mut self, mode: &'static DisplayMode) -> Result<(), G2gError> {
        self.log().video_mode = Some(mode.name);
        Ok(())
    }

    fn schedule_video(&mut self, frame: &ScheduledVideoFrame<'_>) -> Result<(), G2gError> {
        self.log().pictures.push(ScheduledPicture {
            display_time_ns: frame.display_time_ns,
            duration_ns: frame.duration_ns,
            row_bytes: frame.row_bytes,
        });
        Ok(())
    }

    fn enable_audio(&mut self, format: AudioFormat, channels: u8) -> Result<(), G2gError> {
        self.log().audio = Some((format, channels));
        Ok(())
    }

    fn begin_audio_preroll(&mut self) -> Result<(), G2gError> {
        self.log().audio_prerolls += 1;
        Ok(())
    }

    fn schedule_audio(
        &mut self,
        _samples: &[u8],
        sample_frames: u32,
        stream_time_ns: u64,
    ) -> Result<u32, G2gError> {
        let mut log = self.log();
        let taken = log
            .audio_frames_per_call
            .map_or(sample_frames, |most| most.min(sample_frames));
        log.samples.push(ScheduledSamples {
            stream_time_ns,
            sample_frames: taken,
        });
        Ok(taken)
    }

    fn start_playback(&mut self, start_time_ns: u64) -> Result<(), G2gError> {
        self.log().playback_starts.push(start_time_ns);
        Ok(())
    }

    fn stop_playback(&mut self, stop_time_ns: u64) -> Result<(), G2gError> {
        self.log().playback_stops.push(stop_time_ns);
        Ok(())
    }

    fn hardware_time_ns(&mut self) -> Result<u64, G2gError> {
        Ok((self.time)())
    }
}

pub(crate) fn mock_opener(log: SharedLog, time: BoardTime) -> DeckLinkOpener {
    Arc::new(move |_device_number| {
        Ok(Box::new(MockBoard {
            log: Arc::clone(&log),
            time: Arc::clone(&time),
        }) as Box<dyn DeckLinkOutput>)
    })
}

#[derive(Debug, Default)]
pub(crate) struct ManualClock(AtomicU64);

impl ManualClock {
    pub(crate) fn set(&self, ns: u64) {
        self.0.store(ns, Ordering::Relaxed);
    }
}

impl PipelineClock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

pub(crate) fn manual_board_time(clock: &Arc<ManualClock>) -> BoardTime {
    let clock = Arc::clone(clock);
    Arc::new(move || clock.now_ns())
}

pub(crate) struct NullOutput;

impl OutputSink for NullOutput {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

pub(crate) fn buffer(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..Default::default()
        },
        sequence: 0,
        meta: Default::default(),
    })
}

pub(crate) fn video_caps(format: RawVideoFormat, mode: &DisplayMode, interlace: Interlace) -> Caps {
    let framerate = g2g_core::caps::framerate_q16_from_fraction(
        mode.framerate_numerator,
        mode.framerate_denominator,
    )
    .expect("mode framerate fits Q16");
    Caps::RawVideo {
        format,
        width: Dim::Fixed(mode.width),
        height: Dim::Fixed(mode.height),
        framerate: Rate::Fixed(framerate),
        interlace,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

pub(crate) fn picture_bytes(format: RawVideoFormat, mode: &DisplayMode) -> Vec<u8> {
    let stride = format.row_stride(mode.width).expect("packed format");
    vec![0; stride as usize * mode.height as usize]
}

pub(crate) fn pcm_caps(format: AudioFormat, channels: u8) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate: 48_000,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

// where the Desktop Video installers put the driver library
const DRIVER_DIRECTORIES: [&str; 4] = [
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
];

pub(crate) fn driver_installed() -> bool {
    DRIVER_DIRECTORIES
        .iter()
        .any(|directory| Path::new(directory).join("libDeckLinkAPI.so").exists())
}

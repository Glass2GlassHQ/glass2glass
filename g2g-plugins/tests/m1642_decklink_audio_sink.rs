#![cfg(all(target_os = "linux", feature = "decklink"))]

use std::sync::{Arc, Mutex};

use g2g_core::{
    AsyncElement, AudioFormat, ClockSync, G2gError, Interlace, PropValue, RawVideoFormat,
};
use g2g_plugins::decklink::{display_mode, DisplayMode};
use g2g_plugins::decklinkaudiosink::DeckLinkAudioSink;
use g2g_plugins::decklinkvideosink::DeckLinkVideoSink;

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::{
    buffer, manual_board_time, mock_opener, pcm_caps, picture_bytes, video_caps, BoardLog,
    ManualClock, NullOutput, ScheduledSamples, SharedLog,
};

const MILLISECOND: u64 = 1_000_000;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const SAMPLE_RATE: u64 = 48_000;
// the first buffer on the device is due 200 ms after it arrives
const PREROLL_NS: u64 = 200 * MILLISECOND;
const BOARD_START_NS: u64 = 1_000_000 * MILLISECOND;
const ELECTED_START_NS: u64 = 5_000 * MILLISECOND;
// more than a frame, so a picture timed from its own arrival would land a frame late
const VIDEO_LAG_NS: u64 = 40 * MILLISECOND;
const STEREO: u8 = 2;
const S16_STEREO_BYTES: usize = 4;

struct Device {
    video: DeckLinkVideoSink,
    audio: DeckLinkAudioSink,
    log: SharedLog,
    board: Arc<ManualClock>,
    elected: Arc<ManualClock>,
}

// the audio sink is configured first, before its video sink has opened the device
fn device(device_number: u32, mode: &DisplayMode) -> Device {
    let log: SharedLog = Arc::new(Mutex::new(BoardLog::default()));
    let board = Arc::new(ManualClock::default());
    board.set(BOARD_START_NS);
    let elected = Arc::new(ManualClock::default());
    elected.set(ELECTED_START_NS);
    let mut video = DeckLinkVideoSink::with_opener(
        device_number,
        mock_opener(Arc::clone(&log), manual_board_time(&board)),
    );
    let mut audio = DeckLinkAudioSink::new();
    audio
        .set_property("device-number", PropValue::Int(i64::from(device_number)))
        .expect("device-number");
    audio
        .configure_pipeline(&pcm_caps(AudioFormat::PcmS16Le, STEREO))
        .expect("configure audio");
    video
        .configure_pipeline(&video_caps(
            RawVideoFormat::Uyvy,
            mode,
            Interlace::Progressive,
        ))
        .expect("configure video");
    let sync = ClockSync::new(elected.clone(), ELECTED_START_NS);
    video.set_clock_sync(sync.clone());
    audio.set_clock_sync(sync);
    Device {
        video,
        audio,
        log,
        board,
        elected,
    }
}

#[tokio::test]
async fn audio_follows_each_video_frame_with_its_share_of_samples() {
    let mode = display_mode("1080p2997").expect("mode");
    let mut device = device(164201, mode);
    // 29.97 fps carries 8008 samples in every five frames
    let samples_per_frame: [u32; 5] = [1602, 1601, 1602, 1601, 1602];

    let mut samples_so_far = 0;
    for (index, sample_frames) in samples_per_frame.iter().enumerate() {
        let video_pts_ns = index as u64 * NANOS_PER_SECOND * 1001 / 30_000;
        let audio_pts_ns = samples_so_far * NANOS_PER_SECOND / SAMPLE_RATE;
        // the first buffers arrive at the clocks' start, later ones once they are due, so only the first wait
        let arrival_ns = PREROLL_NS + video_pts_ns;
        if index > 0 {
            device.board.set(BOARD_START_NS + arrival_ns);
            device.elected.set(ELECTED_START_NS + arrival_ns);
        }
        let pcm = vec![0; *sample_frames as usize * S16_STEREO_BYTES];
        device
            .audio
            .process(buffer(pcm, audio_pts_ns), &mut NullOutput)
            .await
            .expect("audio");
        if index == 0 {
            device.board.set(BOARD_START_NS + VIDEO_LAG_NS);
            device.elected.set(ELECTED_START_NS + VIDEO_LAG_NS);
        }
        let picture = picture_bytes(RawVideoFormat::Uyvy, mode);
        device
            .video
            .process(buffer(picture, video_pts_ns), &mut NullOutput)
            .await
            .expect("video");
        samples_so_far += u64::from(*sample_frames);
    }

    let log = device.log.lock().expect("log");
    assert_eq!(log.audio, Some((AudioFormat::PcmS16Le, STEREO)));
    let mut samples_so_far = 0;
    let expected: Vec<ScheduledSamples> = samples_per_frame
        .iter()
        .map(|sample_frames| {
            let pts_ns = samples_so_far * NANOS_PER_SECOND / SAMPLE_RATE;
            samples_so_far += u64::from(*sample_frames);
            ScheduledSamples {
                stream_time_ns: BOARD_START_NS + PREROLL_NS + pts_ns,
                sample_frames: *sample_frames,
            }
        })
        .collect();
    assert_eq!(log.samples, expected);
    // audio reached the board before playback started, and video started it once
    assert_eq!(log.audio_prerolls, 1);
    assert_eq!(log.playback_starts.len(), 1);
    // the first picture came later than the first samples and still lines up with them
    let frame_ns = mode.frame_duration_ns();
    let display_times: Vec<u64> = log.pictures.iter().map(|p| p.display_time_ns).collect();
    let expected: Vec<u64> = (0..samples_per_frame.len() as u64)
        .map(|index| {
            let due_ns = BOARD_START_NS + PREROLL_NS + index * NANOS_PER_SECOND * 1001 / 30_000;
            due_ns.div_ceil(frame_ns) * frame_ns
        })
        .collect();
    assert_eq!(display_times, expected);
}

#[tokio::test]
async fn samples_a_full_board_turns_away_are_offered_again_later_in_the_stream() {
    let mode = display_mode("1080p25").expect("mode");
    let mut device = device(164202, mode);
    let accepted_per_call = 1000;
    device.log.lock().expect("log").audio_frames_per_call = Some(accepted_per_call);

    let sample_frames = 1920;
    let pcm = vec![0; sample_frames as usize * S16_STEREO_BYTES];
    device
        .audio
        .process(buffer(pcm, 0), &mut NullOutput)
        .await
        .expect("audio");

    let first_ns = BOARD_START_NS + PREROLL_NS;
    let rest_offset_ns = u64::from(accepted_per_call) * NANOS_PER_SECOND / SAMPLE_RATE;
    assert_eq!(
        device.log.lock().expect("log").samples,
        vec![
            ScheduledSamples {
                stream_time_ns: first_ns,
                sample_frames: accepted_per_call,
            },
            ScheduledSamples {
                stream_time_ns: first_ns + rest_offset_ns,
                sample_frames: sample_frames - accepted_per_call,
            },
        ]
    );
}

#[test]
fn an_audio_sink_without_a_video_sink_on_its_device_fails_at_configure() {
    let mut audio = DeckLinkAudioSink::new();
    audio
        .set_property("device-number", PropValue::Int(164203))
        .expect("device-number");
    assert_eq!(
        audio
            .configure_pipeline(&pcm_caps(AudioFormat::PcmS16Le, STEREO))
            .err(),
        Some(G2gError::NotConfigured)
    );
}

#[test]
fn audio_outside_the_board_formats_is_refused() {
    let audio = DeckLinkAudioSink::new();
    assert_eq!(
        audio
            .intercept_caps(&pcm_caps(AudioFormat::PcmS16Le, 6))
            .err(),
        Some(G2gError::CapsMismatch)
    );
    assert_eq!(
        audio
            .intercept_caps(&pcm_caps(AudioFormat::PcmF32Le, STEREO))
            .err(),
        Some(G2gError::CapsMismatch)
    );
}

#![cfg(all(target_os = "linux", feature = "decklink"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ClockPriority, ClockSync, Dim, FrameTiming, G2gError,
    HardwareError, Interlace, MonotonicClock, PipelineClock, PipelinePacket, PropError, PropValue,
    RawVideoFormat,
};
use g2g_plugins::decklink::{display_mode, DisplayMode, DISPLAY_MODES};
use g2g_plugins::decklinkvideosink::DeckLinkVideoSink;

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::{
    buffer, driver_installed, manual_board_time, mock_opener, picture_bytes, video_caps, BoardLog,
    BoardTime, ManualClock, NullOutput, SharedLog,
};

const MILLISECOND: u64 = 1_000_000;
// the sink holds a buffer until 30 ms before it is due, and the first is due 200 ms after it arrives
const SCHEDULE_LEAD_NS: u64 = 30 * MILLISECOND;
const PREROLL_NS: u64 = 200 * MILLISECOND;
// a whole number of 25 fps frames
const BOARD_START_NS: u64 = 1_000_000 * MILLISECOND;
const ELECTED_START_NS: u64 = 5_000 * MILLISECOND;

fn mode(name: &str) -> &'static DisplayMode {
    display_mode(name).expect("a GStreamer mode name")
}

// the first buffer arrives at the clocks' start and fixes when running time zero is due, later
// ones arrive at the moment the sink hands them over, so only the first waits
fn arrival_ns(index: u64, pts_ns: u64) -> u64 {
    match index {
        0 => 0,
        _ => PREROLL_NS - SCHEDULE_LEAD_NS + pts_ns,
    }
}

fn sink_on_mock(device_number: u32, time: BoardTime) -> (DeckLinkVideoSink, SharedLog) {
    let log: SharedLog = Arc::new(Mutex::new(BoardLog::default()));
    let sink = DeckLinkVideoSink::with_opener(device_number, mock_opener(Arc::clone(&log), time));
    (sink, log)
}

#[tokio::test]
async fn frames_are_scheduled_at_their_stream_times_and_play_out_at_end_of_stream() {
    let board = Arc::new(ManualClock::default());
    let elected = Arc::new(ManualClock::default());
    let (mut sink, log) = sink_on_mock(164101, manual_board_time(&board));
    let mode = mode("1080p25");
    let frame_ns = mode.frame_duration_ns();
    sink.configure_pipeline(&video_caps(
        RawVideoFormat::Uyvy,
        mode,
        Interlace::Progressive,
    ))
    .expect("configure");
    sink.set_clock_sync(ClockSync::new(elected.clone(), ELECTED_START_NS));

    let frames = 5;
    for index in 0..frames {
        let pts_ns = index * frame_ns;
        board.set(BOARD_START_NS + arrival_ns(index, pts_ns));
        elected.set(ELECTED_START_NS + arrival_ns(index, pts_ns));
        let packet = buffer(picture_bytes(RawVideoFormat::Uyvy, mode), pts_ns);
        sink.process(packet, &mut NullOutput).await.expect("frame");
    }

    let first_display_ns = BOARD_START_NS + PREROLL_NS;
    let end_ns = first_display_ns + frames * frame_ns;
    board.set(end_ns);
    sink.process(PipelinePacket::Eos, &mut NullOutput)
        .await
        .expect("eos");

    let log = log.lock().expect("log");
    assert_eq!(log.video_mode, Some("1080p25"));
    let display_times: Vec<u64> = log.pictures.iter().map(|p| p.display_time_ns).collect();
    let expected: Vec<u64> = (0..frames)
        .map(|index| first_display_ns + index * frame_ns)
        .collect();
    assert_eq!(display_times, expected);
    assert!(log.pictures.iter().all(|p| p.duration_ns == frame_ns));
    assert_eq!(
        log.pictures[0].row_bytes,
        RawVideoFormat::Uyvy.row_stride(mode.width).expect("packed")
    );
    assert_eq!(log.playback_starts, vec![BOARD_START_NS]);
    assert_eq!(log.playback_stops, vec![end_ns]);
}

#[tokio::test]
async fn display_times_land_on_the_frame_grid_of_a_fractional_rate() {
    let board = Arc::new(ManualClock::default());
    let elected = Arc::new(ManualClock::default());
    let (mut sink, log) = sink_on_mock(164102, manual_board_time(&board));
    let mode = mode("1080p2997");
    let frame_ns = mode.frame_duration_ns();
    sink.configure_pipeline(&video_caps(
        RawVideoFormat::V210,
        mode,
        Interlace::Progressive,
    ))
    .expect("configure");
    sink.set_clock_sync(ClockSync::new(elected.clone(), ELECTED_START_NS));

    for index in 0..4u64 {
        let pts_ns = index * 1_000_000_000 * 1001 / 30_000;
        board.set(BOARD_START_NS + arrival_ns(index, pts_ns));
        elected.set(ELECTED_START_NS + arrival_ns(index, pts_ns));
        let packet = buffer(picture_bytes(RawVideoFormat::V210, mode), pts_ns);
        sink.process(packet, &mut NullOutput).await.expect("frame");
    }

    let log = log.lock().expect("log");
    let display_times: Vec<u64> = log.pictures.iter().map(|p| p.display_time_ns).collect();
    assert!(display_times.iter().all(|t| t % frame_ns == 0));
    assert!(display_times
        .windows(2)
        .all(|pair| pair[1] - pair[0] == frame_ns));
    assert!(display_times[0] >= BOARD_START_NS + PREROLL_NS);
    assert!(display_times[0] < BOARD_START_NS + PREROLL_NS + frame_ns);
}

#[tokio::test]
async fn the_offered_clock_follows_the_board_clock() {
    // a board clock running 1% fast against the host's monotonic clock
    let board_rate = 1.01;
    let origin = MonotonicClock.now_ns();
    let time: BoardTime = Arc::new(move || {
        BOARD_START_NS + ((MonotonicClock.now_ns() - origin) as f64 * board_rate) as u64
    });
    let (mut sink, log) = sink_on_mock(164103, time);
    let mode = mode("720p50");
    sink.configure_pipeline(&video_caps(
        RawVideoFormat::Bgra8,
        mode,
        Interlace::Progressive,
    ))
    .expect("configure");

    let candidate = sink
        .provide_clock()
        .expect("the sink offers the board clock");
    assert_eq!(candidate.priority, ClockPriority::AudioProvider);
    let clock = sink.clock();
    sink.set_clock_sync(ClockSync::new(
        candidate.clock.clone(),
        candidate.clock.now_ns(),
    ));

    let frames = 12;
    for _ in 0..frames {
        let packet = buffer(
            picture_bytes(RawVideoFormat::Bgra8, mode),
            FrameTiming::PTS_NONE,
        );
        sink.process(packet, &mut NullOutput).await.expect("frame");
        std::thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(log.lock().expect("log").pictures.len(), frames);
    assert_eq!(clock.observations(), frames);
    assert!(
        (clock.slope() - board_rate).abs() < 1e-3,
        "slope {} for a board at {board_rate}",
        clock.slope()
    );
}

#[test]
fn auto_mode_picks_the_mode_the_caps_describe() {
    let cases = [
        ("720p50", Interlace::Progressive, 164110),
        ("1080p25", Interlace::Progressive, 164111),
        ("1080i50", Interlace::Interleaved, 164112),
        ("ntsc", Interlace::Interleaved, 164113),
    ];
    for (name, interlace, device_number) in cases {
        let (mut sink, log) = sink_on_mock(device_number, manual_board_time(&Arc::default()));
        let caps = video_caps(RawVideoFormat::Uyvy, mode(name), interlace);
        sink.configure_pipeline(&caps).expect("configure");
        assert_eq!(log.lock().expect("log").video_mode, Some(name));
    }
}

#[test]
fn caps_without_a_mode_fail_at_configure() {
    let (mut sink, log) = sink_on_mock(164120, manual_board_time(&Arc::default()));
    let Caps::RawVideo {
        format,
        framerate,
        interlace,
        colorimetry,
        ..
    } = video_caps(
        RawVideoFormat::Uyvy,
        mode("1080p25"),
        Interlace::Progressive,
    )
    else {
        unreachable!("video_caps builds raw video caps");
    };
    let odd_size = Caps::RawVideo {
        format,
        width: Dim::Fixed(1000),
        height: Dim::Fixed(1000),
        framerate,
        interlace,
        colorimetry,
    };
    assert_eq!(
        sink.configure_pipeline(&odd_size).err(),
        Some(G2gError::CapsMismatch)
    );
    assert_eq!(log.lock().expect("log").video_mode, None);
}

#[test]
fn an_explicit_mode_rejects_caps_of_another_mode() {
    let (mut sink, log) = sink_on_mock(164130, manual_board_time(&Arc::default()));
    sink.set_property("mode", PropValue::Str("1080p25".into()))
        .expect("mode");
    let other = video_caps(RawVideoFormat::Uyvy, mode("720p50"), Interlace::Progressive);
    assert_eq!(
        sink.configure_pipeline(&other).err(),
        Some(G2gError::CapsMismatch)
    );
    assert_eq!(log.lock().expect("log").video_mode, None);

    let CapsConstraint::Accepts(accepted) = sink.caps_constraint_as_sink() else {
        panic!("a sink constraint");
    };
    let mode = mode("1080p25");
    assert!(accepted.alternatives().iter().all(|caps| matches!(
        caps,
        Caps::RawVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. }
            if *w == mode.width && *h == mode.height
    )));

    let matching = video_caps(RawVideoFormat::R210, mode, Interlace::Progressive);
    sink.configure_pipeline(&matching).expect("configure");
    assert_eq!(log.lock().expect("log").video_mode, Some("1080p25"));
}

#[test]
fn a_second_video_sink_on_one_device_fails_at_configure() {
    let board = manual_board_time(&Arc::default());
    let (mut first, _) = sink_on_mock(164140, Arc::clone(&board));
    let (mut second, _) = sink_on_mock(164140, board);
    let caps = video_caps(
        RawVideoFormat::Uyvy,
        mode("1080p25"),
        Interlace::Progressive,
    );
    first.configure_pipeline(&caps).expect("configure");
    assert!(matches!(
        second.configure_pipeline(&caps),
        Err(G2gError::Hardware(_))
    ));
}

#[test]
fn the_mode_property_lists_every_mode_and_refuses_others() {
    let (mut sink, _) = sink_on_mock(164150, manual_board_time(&Arc::default()));
    let spec = sink
        .properties()
        .iter()
        .find(|spec| spec.name == "mode")
        .expect("mode property");
    let listed: Vec<&str> = spec.enum_nicks().collect();
    let expected: Vec<&str> = core::iter::once("auto")
        .chain(DISPLAY_MODES.iter().map(|mode| mode.name))
        .collect();
    assert_eq!(listed, expected);

    assert_eq!(
        sink.set_property("mode", PropValue::Str("ntsc-widescreen".into())),
        Err(PropError::Value)
    );
    sink.set_property("mode", PropValue::Str("2160p5994".into()))
        .expect("mode");
    assert_eq!(
        sink.get_property("mode"),
        Some(PropValue::Str("2160p5994".into()))
    );
    sink.set_property("device-number", PropValue::Int(164151))
        .expect("device-number");
    assert_eq!(
        sink.get_property("device-number"),
        Some(PropValue::Int(164151))
    );
}

#[test]
fn the_sink_reports_a_missing_driver_library_at_configure() {
    if driver_installed() {
        eprintln!("skip m1641: the driver library is installed on this host");
        return;
    }
    let mut sink = DeckLinkVideoSink::new();
    sink.set_property("device-number", PropValue::Int(164160))
        .expect("device-number");
    let caps = video_caps(
        RawVideoFormat::Uyvy,
        mode("1080p25"),
        Interlace::Progressive,
    );
    assert_eq!(
        sink.configure_pipeline(&caps).err(),
        Some(G2gError::Hardware(HardwareError::LibraryNotFound(
            "libDeckLinkAPI.so"
        )))
    );
}
